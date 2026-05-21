from __future__ import print_function
import os
import sys
import numpy as np
import torch
import random
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pdb
import time
from torch_scatter import scatter_softmax
sys.path.append('%s/lib' % os.path.dirname(os.path.realpath(__file__)))
from gnn_lib import GNNLIB
from pytorch_util import weights_init

from torch_scatter import scatter_softmax, scatter_add


''' Random masking part weights '''
class SMA_RMLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, edge_dropout_rate=0.2, heads=1):
        super(SMA_RMLayer, self).__init__()
        self.dropout = dropout
        self.edge_dropout_rate = edge_dropout_rate
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.heads = heads                  # 多头
        self.hdim = out_features // heads   # 每个头的维度（必须整除）

        # === 多头投影 ===
        self.W = nn.Parameter(torch.empty(in_features, heads * self.hdim))
        self.a = nn.Parameter(torch.empty(2 * self.hdim, heads))
        self.symbol_proj = nn.Linear(1, heads)  # 符号感知可学习层

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)


        self.prelu_attn = nn.PReLU(num_parameters=1, init=0.0)
        for param in self.prelu_attn.parameters():
            param.requires_grad = False  


        self.prelu_out = nn.PReLU(num_parameters=1, init=0.0)
        for param in self.prelu_out.parameters():
            param.requires_grad = False  

    def forward(self, h, adj_sparse, symbol_feat=None):
        H = self.heads
        D = self.hdim
        N = h.size(0)

       
        Wh = torch.matmul(h, self.W).view(N, H, D)

    
        edge_index = adj_sparse._indices()
        src, dst = edge_index

     
        if self.training and self.edge_dropout_rate > 0:
            mask = F.dropout(torch.ones(src.size(0), device=h.device), p=self.edge_dropout_rate, training=self.training)
            keep = mask > 0
            src = src[keep]
            dst = dst[keep]

        # 多头注意力
        Wh_s = Wh[src]
        Wh_d = Wh[dst]
        a_input = torch.cat([Wh_s, Wh_d], dim=-1)
        e = torch.matmul(a_input, self.a).squeeze(-1)

        e = self.prelu_attn(e)

        # ===================== 【论文核心】Symbol-aware 注入 =====================
        if symbol_feat is not None:
            sym = (symbol_feat[src] + symbol_feat[dst]).float().unsqueeze(-1)
            e = e + self.symbol_proj(sym)

        # 多头稀疏 softmax
        attn = torch.stack([scatter_softmax(e[:, i], src, dim=0) for i in range(H)], dim=1)
        attn = F.dropout(attn, self.dropout, training=self.training)

        out = torch.zeros(N, H, D, device=h.device)
        for i in range(H):
            out[:, i] = scatter_add(attn[:, i:i+1] * Wh[dst, i], src, dim=0, dim_size=N)

        h_prime = out.flatten(1) if self.concat else out.mean(1)
        
        return self.prelu_out(h_prime) if self.concat else h_prime



class Readout(nn.Module):
    def __init__(self, in_features):
        super(Readout, self).__init__()
        self.in_features = in_features
        # 移除注意力权重层（不再需要）
        # self.attention_weights = nn.Linear(in_features, 1)

    def forward(self, x):
        stacked_x = torch.stack(x, dim=0)  # (num_layers, num_nodes_in_batch, layer_output_dim)
        
        # 直接转置并拼接所有层的特征，不进行注意力加权
        transposed_features = stacked_x.permute(1, 0, 2)  # (num_nodes_in_batch, num_layers, layer_output_dim)
        fused_features_concatenated = transposed_features.reshape(
            transposed_features.size(0), -1  # (num_nodes_in_batch, num_layers * layer_output_dim)
        )
        
        return fused_features_concatenated


class SMA(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats, latent_dim=[32, 32, 32, 1], conv1d_channels=[16, 32], conv1d_kws=[0, 5], conv1d_activation='PReLU', edge_dropout_rate=0.2,multiplexing_count=1  ):
        super(SMA, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.total_latent_dim = latent_dim[-1] if len(latent_dim) > 0 else 0
        self.edge_dropout_rate = edge_dropout_rate # 保存边缘 dropout 率

        first_layer_input_dim = num_node_feats + num_edge_feats

        self.LN = nn.ModuleList()
       
        self.conv_params = nn.ModuleList()
        # first layer  
        self.conv_params.append(SMA_RMLayer(first_layer_input_dim, latent_dim[0], dropout=0.2, alpha=0.3, concat=True, edge_dropout_rate=self.edge_dropout_rate))
        self.LN.append( nn.LayerNorm(latent_dim[0]) )
        # subsequent layer 
        for i in range(1, len(latent_dim)):
            self.conv_params.append(SMA_RMLayer(latent_dim[i-1], latent_dim[i], dropout=0.2, alpha=0.3, concat=True, edge_dropout_rate=self.edge_dropout_rate))
            self.LN.append( nn.LayerNorm(latent_dim[i]))

        # *** 关键修改：更新 dense_dim 以反映 Readout 新的输出维度 ***
        # Readout 的输出维度是 num_layers * latent_dim[-1]
        # 然后在 out_emb 步骤中又将两个这样的向量拼接，所以是 2 * (num_layers * latent_dim[-1])
        self.demux_head = Parameter(torch.eye(multiplexing_count), requires_grad=True).cuda()
        self.dense_dim = 2 * len(self.latent_dim) * latent_dim[-1] +  2*self.demux_head.shape[0]

        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)
        self.BN = nn.BatchNorm1d(self.dense_dim) # BatchNorm1d 的输入维度也需匹配

        self.conv1d_activation = nn.PReLU(init=0.0)
        self.conv1d_activation.weight.requires_grad = False
        # 新增 Readout 模块，in_features 依然是单层输出维度
        self.readout = Readout(latent_dim[-1])

        weights_init(self)
        self.totalnum1=0
        self.comput_num=0

    def forward(self, graph_list, node_feat, edge_feat, epoch):
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        node_degs = [torch.Tensor(graph_list[i].degs) + 1 for i in range(len(graph_list))]
        node_degs = torch.cat(node_degs).unsqueeze(1)
        subgraph2list = [graph_list[i].subgraph2list for i in range(len(graph_list))]
        len_out = [graph_list[i].len_out for i in range(len(graph_list))]

        n2n_sp, e2n_sp, subg_sp = GNNLIB.PrepareSparseMatrices(graph_list)
        if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
            n2n_sp = n2n_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()
            node_degs = node_degs.cuda()

        h = self.embedding(node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs, subgraph2list, len_out, epoch)

        return h


    def embedding(self, node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs, subgraph2list, len_out, epoch):
        # if exists edge feature, concatenate to node feature vector 
        if edge_feat is not None:
            e2npool_input = torch.sparse.mm(e2n_sp, edge_feat)
            node_feat = torch.cat([node_feat, e2npool_input], 1)
        self.totalnum1 += n2n_sp.to_dense().shape[0] 
        self.comput_num += 1

        # graph convolution layers 
        lv = 0
        cur_message_layer = node_feat

        cat_message_layers = []

        while lv < len(self.latent_dim):
            cur_message_layer = self.conv_params[lv](cur_message_layer, n2n_sp)
            cur_message_layer = self.LN[lv](cur_message_layer)
            cur_message_layer = self.conv1d_activation(cur_message_layer)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        fused_features = self.readout(cat_message_layers)

        
        ''' demultiplexing from the fused_outputs '''
        out_emb = []
        demux_idx=0
        for i, j in subgraph2list[0]:    
            feat1 = torch.cat([self.demux_head[demux_idx] , fused_features[i]] , dim=0) 
            feat2 = torch.cat([self.demux_head[demux_idx] , fused_features[j]] , dim=0) 
            
            out_emb.append(torch.cat([feat1,feat2],dim=0))
            demux_idx= demux_idx+1
                    
        to_dense = torch.stack(out_emb)            
        
        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            return out_linear 
        else:
            return to_dense