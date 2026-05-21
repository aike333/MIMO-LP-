from __future__ import print_function
import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
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
from pytorch_util import weights_init, gnn_spmm


# ===================== Random Masking Layer ｜全部改用可学习PReLU =====================
class RMLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, edge_dropout_rate=0.0):
        super(RMLayer, self).__init__()
        self.dropout = dropout
        self.edge_dropout_rate = edge_dropout_rate
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.prelu = nn.PReLU()

    def forward(self, h, adj_sparse):
        Wh = torch.matmul(h, self.W)
        edge_index = adj_sparse._indices()
        original_num_edges = edge_index.size(1)

        if self.training and self.edge_dropout_rate > 0:
            edge_mask = torch.ones(original_num_edges, device=h.device)
            edge_mask = F.dropout(edge_mask, p=self.edge_dropout_rate, training=self.training)
            kept_edges_indices = edge_mask.nonzero(as_tuple=True)[0]

            edge_index_filtered = edge_index[:, kept_edges_indices]
            edge_src_filtered = edge_index_filtered[0]
            edge_dst_filtered = edge_index_filtered[1]

            a_input = torch.cat([Wh[edge_src_filtered], Wh[edge_dst_filtered]], dim=1)
        else:
            edge_src_filtered = edge_index[0]
            edge_dst_filtered = edge_index[1]
            edge_index_filtered = edge_index
            a_input = torch.cat([Wh[edge_src_filtered], Wh[edge_dst_filtered]], dim=1)

        e = self.prelu(torch.matmul(a_input, self.a).squeeze(1))
        attention_values = scatter_softmax(e, edge_src_filtered, dim=0)
        attention_values = F.dropout(attention_values, self.dropout, training=self.training)

        attention = torch.sparse_coo_tensor(edge_index_filtered, attention_values, adj_sparse.size(), device=h.device)
        h_prime = torch.sparse.mm(attention, Wh)

     
        return h_prime


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], output_dim=None):
        super(MLP, self).__init__()
        # 全部替换为可学习PReLU
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.PReLU())
        
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.PReLU())
        
        if output_dim is not None:
            layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.mlp(x)


class NCN(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats, latent_dim=[32, 32, 32, 1],
                 conv1d_channels=[16, 32], conv1d_kws=[0, 5],conv1d_activation='PReLU',
                 edge_dropout_rate=0.2, multiplexing_count=1):
        super(NCN, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.total_latent_dim = sum(latent_dim)
        self.edge_dropout_rate = edge_dropout_rate
        self.mux_count = multiplexing_count

        self.LN = nn.ModuleList()
        self.conv_params = nn.ModuleList()
        # 每层单独定义 可学习PReLU
        self.layer_prelus = nn.ModuleList()

        first_dim = num_node_feats + num_edge_feats
        self.conv_params.append(RMLayer(first_dim, latent_dim[0], dropout=0.1, alpha=0.3,
                                        concat=True, edge_dropout_rate=edge_dropout_rate))
        self.LN.append(nn.LayerNorm(latent_dim[0]))
        self.layer_prelus.append(nn.PReLU())

        for i in range(1, len(latent_dim)):
            self.conv_params.append(RMLayer(latent_dim[i-1], latent_dim[i], dropout=0.1, alpha=0.3,
                                            concat=True, edge_dropout_rate=edge_dropout_rate))
            self.LN.append(nn.LayerNorm(latent_dim[i]))
            self.layer_prelus.append(nn.PReLU())

        self.dense_dim = (len(latent_dim)*latent_dim[-1])*2
        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)
        self.BN = nn.BatchNorm1d(self.dense_dim)

        # 全局激活：可学习PReLU
        self.conv1d_activation = nn.PReLU()
        weights_init(self)

        self.totalnum1=0
        self.comput_num=0
        self.demux_head = Parameter(torch.eye(multiplexing_count), requires_grad=True)

        single_feat_dim = self.total_latent_dim + multiplexing_count
        concat_dim = 2 * single_feat_dim
        self.transform_mlp = MLP(
            input_dim=concat_dim,
            hidden_dims=[2048],
            output_dim=2 * self.total_latent_dim
        )

    def forward(self, graph_list, node_feat, edge_feat, epoch):
        graph_sizes = [g.num_nodes for g in graph_list]
        node_degs = [torch.Tensor(g.degs) + 1 for g in graph_list]
        node_degs = torch.cat(node_degs).unsqueeze(1)
        subgraph2list = [g.subgraph2list for g in graph_list]
        len_out = [g.len_out for g in graph_list]

        n2n_sp, e2n_sp, subg_sp = GNNLIB.PrepareSparseMatrices(graph_list)

        if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
            n2n_sp = n2n_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()
            node_degs = node_degs.cuda()
            self.demux_head = self.demux_head.cuda()

        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat).cuda() if torch.cuda.is_available() else Variable(edge_feat)

        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)
        node_degs = Variable(node_degs)

        h = self.embedding(node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs, subgraph2list, len_out, epoch)
        return h

    def embedding(self, node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs, subgraph2list, len_out, epoch):
        if edge_feat is not None:
            e2npool_input = gnn_spmm(e2n_sp, edge_feat)
            node_feat = torch.cat([node_feat, e2npool_input], 1)

        self.totalnum1 += n2n_sp.to_dense().shape[0]
        self.comput_num += 1

        lv = 0
        cur_message_layer = node_feat
        cat_message_layers = []

        while lv < len(self.latent_dim):
            cur_message_layer = self.conv_params[lv](cur_message_layer, n2n_sp)
            cur_message_layer = self.LN[lv](cur_message_layer)
            # 每层使用独立可学习PReLU
            cur_message_layer = self.layer_prelus[lv](cur_message_layer)
            cur_message_layer = self.conv1d_activation(cur_message_layer)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, 1)
        out_emb = []
        demux_idx = 0

        # 【完全保留】原版 subgraph2list[0] 不修改
        for i, j, common_nodes in subgraph2list[0]:
            feat1 = torch.cat([self.demux_head[demux_idx], cur_message_layer[i]], dim=0)
            feat2 = torch.cat([self.demux_head[demux_idx], cur_message_layer[j]], dim=0)

            hadamard = feat1 * feat2
            common_feat = torch.zeros_like(feat1)
            for node in common_nodes:
                common_feat += torch.cat([self.demux_head[demux_idx], cur_message_layer[node]], dim=0)

            out_emb.append(torch.cat([hadamard, common_feat], dim=0))
            demux_idx += 1

        to_dense = torch.stack(out_emb)
        reluact_fp = self.transform_mlp(to_dense)
        return reluact_fp


