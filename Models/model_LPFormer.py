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


# Given a target link (a, b), LPFormer models the pairwise encoding via an attention module that learns how a and b relate in the context of various LP factors.
class CrossAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, y):
        z = torch.cat([x, y], dim=-1)
        z = self.proj(z)
        return torch.tanh(z)  

''' Random masking part weights '''
class RMLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, edge_dropout_rate=0.4):
        super(RMLayer, self).__init__()
        self.dropout = dropout  
        self.edge_dropout_rate = edge_dropout_rate 
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  
        self.a = Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.prelu_attn = nn.PReLU(num_parameters=1, init=0.0)
        self.prelu_out = nn.PReLU(num_parameters=1, init=0.0)
        
        for param in self.prelu_attn.parameters():
            param.requires_grad = False
        for param in self.prelu_out.parameters():
            param.requires_grad = False

    def forward(self, h, adj_sparse):
        Wh = torch.matmul(h, self.W)  
        num_nodes = Wh.size(0)
        
        edge_index = adj_sparse._indices()  
        original_num_edges = edge_index.size(1)
        
        if self.training and self.edge_dropout_rate > 0:
            edge_mask = torch.ones(original_num_edges, device=h.device)
            edge_mask = F.dropout(edge_mask, p=self.edge_dropout_rate, training=self.training)
            kept_edges_indices = edge_mask.nonzero(as_tuple=True)[0]
            edge_index_filtered = edge_index[:, kept_edges_indices]  
        else:
            edge_index_filtered = edge_index

        src_nodes = edge_index_filtered[0] 
        dst_nodes = edge_index_filtered[1]  
        a_input = torch.cat([Wh[src_nodes], Wh[dst_nodes]], dim=1)  
        
        e = self.prelu_attn(torch.matmul(a_input, self.a).squeeze(1))  

        attention_values = scatter_softmax(e, src_nodes, dim=0)  
        attention_values = F.dropout(attention_values, self.dropout, training=self.training)
        
        attention_matrix = torch.sparse_coo_tensor(
            edge_index_filtered, attention_values,
            size=(num_nodes, num_nodes), device=h.device
        )
        h_prime = torch.sparse.mm(attention_matrix, Wh)  

        return self.prelu_out(h_prime)




class Readout(nn.Module):
    # layer attention 
    def __init__(self, in_features):
        super(Readout, self).__init__()
        self.in_features = in_features
        self.attention_weights = nn.Linear(in_features, 1)

    def forward(self, x):
        stacked_x = torch.stack(x, dim=0) 
        scores = self.attention_weights(stacked_x) 
        alpha = F.softmax(scores, dim=0) 
        weighted_features_per_layer = alpha * stacked_x 
        
        transposed_weighted_features = weighted_features_per_layer.permute(1, 0, 2)
        fused_features_concatenated = transposed_weighted_features.reshape(
            transposed_weighted_features.size(0), -1
        )
        
        return fused_features_concatenated


class LPFormer(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats, latent_dim=[32, 32, 32, 1], conv1d_channels=[16, 32], conv1d_kws=[0, 5], conv1d_activation='PReLU', edge_dropout_rate=0.2,multiplexing_count=1  ):
        super(LPFormer, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.total_latent_dim = latent_dim[-1] if len(latent_dim) > 0 else 0
        self.edge_dropout_rate = edge_dropout_rate 

        first_layer_input_dim = num_node_feats + num_edge_feats
        self.LN = nn.ModuleList()
        self.conv_params = nn.ModuleList()
        self.conv_params.append(RMLayer(first_layer_input_dim, latent_dim[0], dropout=0.2, alpha=0.3, concat=True, edge_dropout_rate=self.edge_dropout_rate))
        self.LN.append( nn.LayerNorm(latent_dim[0]) )
        for i in range(1, len(self.latent_dim)):
            self.conv_params.append(RMLayer(latent_dim[i-1], latent_dim[i], dropout=0.2, alpha=0.3, concat=True, edge_dropout_rate=self.edge_dropout_rate))
            self.LN.append( nn.LayerNorm(latent_dim[i]))

        self.demux_head = Parameter(torch.eye(multiplexing_count), requires_grad=True).cuda()
        self.dense_dim = 2 * len(self.latent_dim) * latent_dim[-1] +  2*self.demux_head.shape[0] +3

    
        attn_dim = latent_dim[-1] * len(latent_dim)
        self.cross_attn = CrossAttention(attn_dim)

        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)
        self.BN = nn.BatchNorm1d(self.dense_dim) 

        self.conv1d_activation = nn.PReLU(init=0.0)
        self.conv1d_activation.weight.requires_grad = False
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
        if edge_feat is not None:
            e2npool_input = torch.sparse.mm(e2n_sp, edge_feat)
            node_feat = torch.cat([node_feat, e2npool_input], 1)
        self.totalnum1 += n2n_sp.to_dense().shape[0] 
        self.comput_num += 1

        cur_message_layer = node_feat
        cat_message_layers = []

        lv = 0
        while lv < len(self.latent_dim):
            cur_message_layer = self.conv_params[lv](cur_message_layer, n2n_sp)
            cur_message_layer = self.LN[lv](cur_message_layer)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        fused_features = self.readout(cat_message_layers)

        out_emb = []
        demux_idx=0
        for i, j , size_arrays in subgraph2list[0]:  
            # 稳定交叉融合，提供有效梯度
            h_i = fused_features[i]
            h_j = fused_features[j]
            cross = self.cross_attn(h_i, h_j)

            # 原版拼接
            all_size =  torch.tensor(size_arrays,dtype=torch.float32, device= self.demux_head[0].device )
            feat1 = torch.cat([self.demux_head[demux_idx] , h_i  ] , dim=0) 
            feat2 = torch.cat([self.demux_head[demux_idx] , h_j  ] , dim=0) 
            out_emb.append(torch.cat([feat1,feat2,all_size],dim=0))
            demux_idx= demux_idx+1
        
        to_dense = torch.stack(out_emb)            
        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            return out_linear 
        else:
            return to_dense