from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from gensim.models import Word2Vec
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/software/node2vec/src' % cur_dir)
from util import GNNGraph
import node2vec
import multiprocessing as mp
from itertools import islice
from scipy.stats import ortho_group
from sklearn.preprocessing import StandardScaler
import scipy.sparse as ssp
from sklearn.decomposition import PCA 
import pandas as pd
from sklearn.cluster import KMeans
np.seterr(all='warn')


from scipy.sparse import csgraph
import csv

""" Negative Sampling for AUC-Evaluated Datasets"""
def sample_neg_in_AUC(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None,
               all_unknown_as_negative=False):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None and test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None and train_pos is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num = len(train_pos[0]) if train_pos else 0
    test_num = len(test_pos[0]) if test_pos else 0
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    if not all_unknown_as_negative:
        # sample a portion unknown links as train_negs and test_negs (no overlap)
        while len(neg[0]) < train_num + test_num:
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg  = (neg[0][:train_num], neg[1][:train_num])
        test_neg = (neg[0][train_num:], neg[1][train_num:])
    else:
        # regard all unknown links as test_negs, sample a portion from them as train_negs
        while len(neg[0]) < train_num:
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg  = (neg[0], neg[1])
        test_neg_i, test_neg_j, _ = ssp.find(ssp.triu(net==0, k=1))
        test_neg = (test_neg_i.tolist(), test_neg_j.tolist())

    return train_pos, train_neg, test_pos, test_neg

""" Negative Sampling for HR@-Evaluated Datasets"""
def sample_neg_in_HR(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None,
               all_unknown_as_negative=False):
    net_triu = ssp.triu(net, k=1)
    row, col, _ = ssp.find(net_triu)
    if train_pos is None and test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    if max_train_num is not None and train_pos is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        perm2 = np.random.permutation(len(test_pos[0]))[:int(max_train_num/4)]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
        test_pos = (test_pos[0][perm2], test_pos[1][perm2])
    
    train_num = len(train_pos[0]) if train_pos else 0
    test_num = len(test_pos[0]) if test_pos else 0
    print("train_num!!!!:",train_num    )
    print("test_num!!!!:",test_num    )
    neg = ([], [])
    n = net.shape[0]
    if not all_unknown_as_negative:
        # sample a portion unknown links as train_negs and test_negs (no overlap)
        while len(neg[0]) < (train_num + test_num*100):
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        
        train_neg  = (neg[0][:int(train_num) ], neg[1][:int(train_num)])
        test_neg = (neg[0][int(train_num):], neg[1][int(train_num):])
     
    else:
        # regard all unknown links as test_negs, sample a portion from them as train_negs
        while len(neg[0]) < train_num:
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg  = (neg[0], neg[1])
        test_neg_i, test_neg_j, _ = ssp.find(ssp.triu(net==0, k=1))
        test_neg = (test_neg_i.tolist(), test_neg_j.tolist())

    return train_pos, train_neg, test_pos, test_neg

""" subgraph extraction  """
def links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, h=3, 
                    max_nodes_per_hop=None, node_information=None, PPR_arrays=None, no_parallel=False,model_name="SEAL",multiplexing_count=1):
        
    # extract enclosing subgraphs
    def subgraph_list(A, links, g_label):
        g_list = []
        link_data = list(zip(links[0], links[1]))

        meg_list = []
        step_len = multiplexing_count
        for step_idx in range(0,len(link_data),step_len): 
            if model_name == "M-SEAL" or model_name == "M-PS2":
                g, n_labels, n_features, subgraph2list,len_out,subgraph_RPE = subgraph_extraction_SEAL(
                    list(link_data[step_idx:step_idx+step_len]), A, h, max_nodes_per_hop, node_information,PPR_arrays
                )
            elif model_name == "M-SMA":
                g, n_labels, n_features, subgraph2list,len_out,subgraph_RPE = subgraph_extraction_SMA(
                    list(link_data[step_idx:step_idx+step_len]), A, h, max_nodes_per_hop, node_information,PPR_arrays
                )                
            elif model_name == "M-LPFormer":
                g, n_labels, n_features, subgraph2list,len_out,subgraph_RPE = subgraph_extraction_LPFormer(
                    list(link_data[step_idx:step_idx+step_len]), A, h, max_nodes_per_hop, node_information,PPR_arrays
                )
            elif model_name == "M-NCN":
                g, n_labels, n_features, subgraph2list,len_out,subgraph_RPE = subgraph_extraction_NCN(
                    list(link_data[step_idx:step_idx+step_len]), A, h, max_nodes_per_hop, node_information,PPR_arrays
                )
            
            g_list.append(GNNGraph(g, list(g_label[step_idx:step_idx+step_len]), n_labels, n_features,subgraph2list,len_out,subgraph_RPE))
            
        return g_list

    print('Subgraph extraction begins...')
    train_graphs, test_graphs = None, None
    pos_per_batch= 1
    neg_per_batch= 1    
    
    if train_pos and train_neg:
        
        train_ind0 = np.concatenate((np.array(train_pos[0]),np.array(train_neg[0])))
        train_ind1 = np.concatenate((np.array(train_pos[1]),np.array(train_neg[1])))
        train_lb = np.concatenate((np.array([1]*len(train_pos[0])),np.array([0]*len(train_neg[0]))))
        
        help_ind = np.arange(len(train_ind0))
        np.random.seed(4234)
        np.random.shuffle(help_ind)    
        train_ind0 = train_ind0[help_ind]
        train_ind1 = train_ind1[help_ind]
        train_lb = train_lb[help_ind]
        
        train_sub_matrix = []
        for i,j in zip(train_ind0, train_ind1):
            subgraph_feature = subgraph_embedding((i,j), A, h, node_information ,max_nodes_per_hop)
            train_sub_matrix.append(subgraph_feature)     
        cluster_labels = cluster_subgraph_features(np.array(train_sub_matrix),800)
        df = pd.DataFrame(cluster_labels)
        df_sorted = df.sort_values(by=0)
        help_ind = list(df_sorted[0].index)
        train_graphs = subgraph_list(A, (train_ind0[help_ind], train_ind1[help_ind] ), train_lb[help_ind] )  
        
    if test_pos and test_neg:
        test_ind0 = np.concatenate((np.array(test_pos[0]),np.array(test_neg[0])))
        test_ind1 = np.concatenate((np.array(test_pos[1]),np.array(test_neg[1])))
        test_lb = np.concatenate((np.array([1]*len(test_pos[0])),np.array([0]*len(test_neg[0]))))
       
        help_ind = np.arange(len(test_ind0))
        np.random.shuffle(help_ind)    
        test_ind0 = test_ind0[help_ind]
        test_ind1 = test_ind1[help_ind]
        test_lb = test_lb[help_ind]        
    
        test_sub_matrix = []
        for i,j in zip(test_ind0, test_ind1):
            subgraph_feature = subgraph_embedding((i,j), A, h, node_information ,max_nodes_per_hop)
            test_sub_matrix.append(subgraph_feature)
        cluster_labels = cluster_subgraph_features(np.array(test_sub_matrix),200)        
        df = pd.DataFrame(cluster_labels)
        df_sorted = df.sort_values(by=0)        
        help_ind = list(df_sorted[0].index)
        
        test_graphs = subgraph_list(A, (test_ind0[help_ind],test_ind1[help_ind]), test_lb[help_ind])  
        
    elif test_pos:
        test_graphs = subgraph_list(A, test_pos, 1)
    return train_graphs, test_graphs


""" The customized subgraph attribute extraction function is designed for the SEAL model.  """
def subgraph_extraction_SEAL(inds, A, h=1, max_nodes_per_hop=None, node_information=None, PPR_arrays=None):

    concat_tag=[]
    concat_nodes=[]
    
    for ind in inds:
        dist = 0
        nodes = set( [])
        visited = set([])
        fringe = set([])
        nodes = nodes.union(set([ind[0], ind[1]])) 
        visited = visited.union(  set([ind[0], ind[1]]))
        fringe = fringe.union(set([ind[0], ind[1]]))    
        for dist in range(1, h+1):
            fringe = neighbors(fringe, A)
            fringe = fringe - visited
            visited = visited.union(fringe)
            if max_nodes_per_hop is not None:
                if max_nodes_per_hop < len(fringe):
                    fringe = random.sample(fringe, max_nodes_per_hop)
            if len(fringe) == 0:
                break
            nodes = nodes.union(fringe)
        if ind[0] in nodes:
            nodes.remove(ind[0])
        if ind[1] in nodes:
            nodes.remove(ind[1])
        nodes = list([ind[0],ind[1]]) + list(nodes)
        subgraph = A[nodes, :][:, nodes]       
        labels = node_label(subgraph)
        node_tag = np.zeros((len(nodes), 100),dtype=int)  #one-hot
        
        node_tag[np.arange(len(nodes)), labels]=1
        concat_tag.append(node_tag)
        concat_nodes.append(nodes)
       
    u_nodes,u_node_tags = multiplexing_subgraph(concat_nodes,concat_tag)    
           
    pnodes = set([])
    for ind in inds:
        pnodes = pnodes.union(set([ind[0], ind[1]]))         
        
    # prepare subgraph featrue for readout function       
    len_out = len(pnodes)
    subgraph2list = []
    for ind in inds:
        # mapping to union graph position
        ind_posA = list(pnodes).index(ind[0])
        ind_posB = list(pnodes).index(ind[1])
       
        subgraph2list.append( (ind_posA,ind_posB) )
    
    temp_embs=[]
    for main_node in pnodes:    
        if main_node in u_nodes:
            ind_pos = list(u_nodes).index(main_node)
            u_nodes.remove(main_node)
            temp_embs.append(u_node_tags[ind_pos])
            del u_node_tags[ind_pos]

    u_nodes = list(pnodes) + list(u_nodes) 
    u_node_tags = temp_embs + u_node_tags
    
    subgraph_RPE = PPR_arrays[u_nodes,0:len_out]    
    subgraph = A[u_nodes, :][:, u_nodes]

    features = None
    if node_information is not None:
        features = node_information[u_nodes]
    # construct nx graph
    g = nx.from_numpy_array(subgraph)
    # remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)
    

    return g, u_node_tags, features, subgraph2list, len_out, subgraph_RPE


def subgraph_extraction_SMA(inds, A, h=1, max_nodes_per_hop=None, node_information=None, PPR_arrays=None):

    concat_tag=[]
    concat_nodes=[]
    
    for ind in inds:
        dist = 0
        nodes = set( [])
        visited = set([])
        fringe = set([])
        nodes = nodes.union(set([ind[0], ind[1]])) 
        visited = visited.union(  set([ind[0], ind[1]]))
        fringe = fringe.union(set([ind[0], ind[1]]))    
        for dist in range(1, h+1):
            fringe = neighbors(fringe, A)
            fringe = fringe - visited
            visited = visited.union(fringe)
            if max_nodes_per_hop is not None:
                if max_nodes_per_hop < len(fringe):
                    fringe = random.sample(fringe, max_nodes_per_hop)
            if len(fringe) == 0:
                break
            nodes = nodes.union(fringe)
        if ind[0] in nodes:
            nodes.remove(ind[0])
        if ind[1] in nodes:
            nodes.remove(ind[1])
        nodes = list([ind[0],ind[1]]) + list(nodes)
        subgraph = A[nodes, :][:, nodes]       
        labels = TADL(subgraph)
        node_tag = np.zeros((len(nodes), 100),dtype=int)  #one-hot
        
        node_tag[np.arange(len(nodes)), labels]=1
        concat_tag.append(node_tag)
        concat_nodes.append(nodes)
       
    u_nodes,u_node_tags = multiplexing_subgraph(concat_nodes,concat_tag)    
           
    pnodes = set([])
    for ind in inds:
        pnodes = pnodes.union(set([ind[0], ind[1]]))         
        
    # prepare subgraph featrue for readout function       
    len_out = len(pnodes)
    subgraph2list = []
    for ind in inds:
        # mapping to union graph position
        ind_posA = list(pnodes).index(ind[0])
        ind_posB = list(pnodes).index(ind[1])
       
        subgraph2list.append( (ind_posA,ind_posB) )
    
    temp_embs=[]
    for main_node in pnodes:    
        if main_node in u_nodes:
            ind_pos = list(u_nodes).index(main_node)
            u_nodes.remove(main_node)
            temp_embs.append(u_node_tags[ind_pos])
            del u_node_tags[ind_pos]

    u_nodes = list(pnodes) + list(u_nodes) 
    u_node_tags = temp_embs + u_node_tags
    
    subgraph_RPE = PPR_arrays[u_nodes,0:len_out]    
    subgraph = A[u_nodes, :][:, u_nodes]

    features = None
    if node_information is not None:
        features = node_information[u_nodes]
    # construct nx graph
    g = nx.from_numpy_array(subgraph)
    # remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)
    

    return g, u_node_tags, features, subgraph2list, len_out, subgraph_RPE



""" The customized subgraph attribute extraction function is designed for the NCN model.  """
def subgraph_extraction_NCN(inds, A, h=1, max_nodes_per_hop=None, node_information=None, PPR_arrays=None):

    concat_tag=[]
    concat_nodes=[]

    for ind in inds:
        dist = 0
        nodes = set( [])
        visited = set([])
        fringe = set([])
        nodes = nodes.union(set([ind[0], ind[1]])) 
        visited = visited.union(  set([ind[0], ind[1]]))
        fringe = fringe.union(set([ind[0], ind[1]]))    
        for dist in range(1, h+1):
            fringe = neighbors(fringe, A)
            fringe = fringe - visited
            visited = visited.union(fringe)
            if max_nodes_per_hop is not None:
                if max_nodes_per_hop < len(fringe):
                    fringe = random.sample(fringe, max_nodes_per_hop)
            if len(fringe) == 0:
                break
            nodes = nodes.union(fringe)
        if ind[0] in nodes:
            nodes.remove(ind[0])
        if ind[1] in nodes:
            nodes.remove(ind[1])
        nodes = list([ind[0],ind[1]]) + list(nodes)
        subgraph = A[nodes, :][:, nodes]      
        labels = node_label(subgraph)
        node_tag = np.zeros((len(nodes), 100),dtype=int)  # one-hot
        node_tag[np.arange(len(nodes)), labels]=1
        concat_tag.append(node_tag  )
        concat_nodes.append(nodes  )
        
    # constructing union graph    
    u_nodes,u_node_tags = multiplexing_subgraph(concat_nodes,concat_tag)    
    
    # queries node pairs 
    pnodes = set([])
    for ind in inds:
        pnodes = pnodes.union(set([ind[0], ind[1]]))         
        
    # prepare subgraph featrue for readout function
    len_out = len(u_nodes)
    subgraph2list = []
    for ind in inds:
        # find intersection
        i_neighbirs, _, _ = ssp.find(A[:, ind[0]])
        j_neighbirs, _, _ = ssp.find(A[:, ind[1]])
        conmon_nodes = set(i_neighbirs) & set(j_neighbirs)
        # mapping to union graph position
        ind_posA = list(u_nodes).index(ind[0])
        ind_posB = list(u_nodes).index(ind[1])
        common_node_pos = []
        for node in conmon_nodes:
            common_node_pos.append(list(u_nodes).index(node))

        subgraph2list.append( (ind_posA,ind_posB,common_node_pos ) )
    
    # reset union graph for prediction
    temp_embs=[]
    for main_node in pnodes:    
        if main_node in u_nodes:
            ind_pos = list(u_nodes).index(main_node)
            u_nodes.remove(main_node)
            temp_embs.append(u_node_tags[ind_pos])
            del u_node_tags[ind_pos]

    u_nodes = list(pnodes) + list(u_nodes) 
    u_node_tags = temp_embs + u_node_tags
    subgraph_RPE = PPR_arrays[u_nodes,0:len_out]    
    
    subgraph = A[u_nodes, :][:, u_nodes]
    features = None
    if node_information is not None:
        features = node_information[u_nodes]

    g = nx.from_numpy_array(subgraph)
    
    # remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)
    
    return g, u_node_tags, features, subgraph2list, len_out, subgraph_RPE

""" The customized subgraph attribute extraction function is designed for the LPFormer model.  """
def subgraph_extraction_LPFormer(inds, A, h=1, max_nodes_per_hop=None,
                                 node_information=None, PPR_arrays=None):

    concat_tag=[]
    concat_nodes=[]
    size_1hop=0
    size_ge_1hop=0

    for ind in inds:
        dist = 0
        nodes = set( [])
        visited = set([])
        fringe = set([])
        nodes = nodes.union(set([ind[0], ind[1]])) 
        visited = visited.union(  set([ind[0], ind[1]]))
        fringe = fringe.union(set([ind[0], ind[1]]))    
        for dist in range(1, h+1):
            fringe = neighbors(fringe, A)
            if dist ==1:
                size_1hop= len(fringe)
            if dist == h+1:
                size_ge_1hop = len( fringe )
            fringe = fringe - visited
            visited = visited.union(fringe)
            if max_nodes_per_hop is not None:
                if max_nodes_per_hop < len(fringe):
                    fringe = random.sample(fringe, max_nodes_per_hop)
            if len(fringe) == 0:
                break
            nodes = nodes.union(fringe)
        if ind[0] in nodes:
            nodes.remove(ind[0])
        if ind[1] in nodes:
            nodes.remove(ind[1])
        nodes = list([ind[0],ind[1]]) + list(nodes)
        subgraph = A[nodes, :][:, nodes]     
        labels = node_label(subgraph)

        node_tag = np.zeros((len(nodes), 100),dtype=int)  # 32 one-hot
        
        node_tag[np.arange(len(nodes)), labels]=1
           
        concat_tag.append(node_tag  )
        concat_nodes.append(nodes  )
       
    u_nodes,u_node_tags = multiplexing_subgraph(concat_nodes,concat_tag)    
           
    pnodes = set([])
    for ind in inds:
        pnodes = pnodes.union(set([ind[0], ind[1]]))         

    # branch3: MIMO-LPFormer
    len_out = len(pnodes)
    subgraph2list = []
    for ind in inds:
        # find intersection
        i_neighbirs, _, _ = ssp.find(A[:, ind[0]])
        j_neighbirs, _, _ = ssp.find(A[:, ind[1]])
        conmon_nodes = set(i_neighbirs) & set(j_neighbirs)
        Size_common = len(conmon_nodes)
        Size_array = [Size_common,size_1hop,size_ge_1hop-size_1hop]
        
        # mapping to union graph position
        ind_posA = list(u_nodes).index(ind[0])
        ind_posB = list(u_nodes).index(ind[1])
        subgraph2list.append( (ind_posA,ind_posB,Size_array))

    temp_embs=[]
    for main_node in pnodes:    
        if main_node in u_nodes:
            ind_pos = list(u_nodes).index(main_node)
            u_nodes.remove(main_node)
            temp_embs.append(u_node_tags[ind_pos])
            del u_node_tags[ind_pos]

    u_nodes = list(pnodes) + list(u_nodes) 
    u_node_tags = temp_embs + u_node_tags
    
    subgraph_RPE = PPR_arrays[u_nodes,0:len_out]    
    subgraph = A[u_nodes, :][:, u_nodes]
    features = None
    if node_information is not None:
        features = node_information[u_nodes]
    # construct nx graph
    g = nx.from_numpy_array(subgraph)
    # remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)

    return g, u_node_tags, features, subgraph2list, len_out, subgraph_RPE


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res


""" local subgraph feature """
def TADL(subgraph):
    """
    修复版：TADL论文标准算法 + 适配SEAL代码维度限制（无越界报错）
    输入：子图邻接矩阵，节点0/1为双锚点
    输出：合规TADL标签，无超大数值，无索引越界
    """
    K = subgraph.shape[0]
    
    # 1. 完整子图计算距离（论文标准，不裁剪）
    dist_matrix = csgraph.shortest_path(subgraph, directed=False, unweighted=True)
    d_u = dist_matrix[:, 0]  # 到锚点0
    d_v = dist_matrix[:, 1]  # 到锚点1

    # 2. 论文核心：全局归一化因子 D_max
    d_sum = d_u + d_v
    d_sum[np.isinf(d_sum)] = 0
    D_max = np.max(d_sum) if d_sum.size > 0 else 0

    # 3. 论文原始公式（安全缩放，不生成超大数）
    min_dist = np.minimum(d_u, d_v)
    total_dist = d_u + d_v
    
    # ✅ 关键修复：限制标签最大值，适配你的维度 size=100
    labels = 1 + min_dist + total_dist
    labels = np.clip(labels, 0, 99)  # 强制限制在 0~99 之间

    # 4. 不可达节点 = 0（和你原代码一致）
    unreachable_mask = np.isinf(d_u) | np.isinf(d_v)
    labels[unreachable_mask] = 0

    # 5. 锚点固定为1（完全兼容原代码）
    labels[0] = 1
    labels[1] = 1
    
    return labels.astype(int)


def node_label(subgraph):
    # subgraph features assignment
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    return labels

""" global node feature  """
def generate_node2vec_embeddings(A, emd_size=128, negative_injection=False, train_neg=None):
    if negative_injection:
        row, col = train_neg
        A = A.copy()
        A[row, col] = 1  # inject negative train
        A[col, row] = 1  # inject negative train
    nx_G = nx.from_numpy_array(A)
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=2)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=10)
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, vector_size=emd_size, window=10, min_count=0, sg=1, 
            workers=8, epochs=1)
    wv = model.wv
    embeddings = np.zeros([A.shape[0], emd_size], dtype='float64')
    sum_embeddings = 0
    empty_list = []
    for i in range(A.shape[0]):
        if str(i) in wv:
            embeddings[i] = wv.word_vec(str(i))
            sum_embeddings += embeddings[i]
        else:
            empty_list.append(i)
    mean_embedding = sum_embeddings / (A.shape[0] - len(empty_list))
    embeddings[empty_list] = mean_embedding
    
    return embeddings

def compute_PPR_viaSim(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized_embeddings = embeddings / norms
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    np.fill_diagonal(similarity_matrix, 1.0)
    
    return similarity_matrix

def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

""" subgraph batching  """
def cluster_subgraph_features(subgraph_feature, n_clusters=3):
   
    if not isinstance(subgraph_feature, np.ndarray):
        subgraph_feature = np.array(subgraph_feature)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(subgraph_feature)
    
    return labels
    
"""  extract the h-hop enclosing subgraph around link 'ind'  """
def subgraph_embedding(ind, A, h=1,node_embeddings=None,max_nodes_per_hop=None):  
    dist = 0
    nodes = set( [])
    visited = set([])
    fringe = set([])
        
    nodes = nodes.union(set([ind[0], ind[1]])) 
    visited = visited.union(  set([ind[0], ind[1]]))
    fringe = fringe.union(set([ind[0], ind[1]]))

    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        
    nodes.remove(ind[0])    
    nodes.remove(ind[1])
    source_nodes=[ind[0] ,ind[1]]
    nodes = source_nodes + list(nodes)         
    
    if node_embeddings is not None:
        features = node_embeddings[nodes]

    subgraph_feat =  features[0] +features[1] 
    return subgraph_feat


''' subgraph multiplexing and union graph construction  ''' 
def multiplexing_subgraph(subgraphs, features):
    
    node_feature_sum = {}
    node_feature_count = {}

    for subgraph, feat_list in zip(subgraphs, features):
        
        for node_id, feat in zip(subgraph, feat_list):
            if node_id in node_feature_sum:
                
                node_feature_sum[node_id] = np.array([
                    a + b for a, b in zip(node_feature_sum[node_id], feat)
                ])
                node_feature_count[node_id] += 1
            else:
                node_feature_sum[node_id] = feat
                node_feature_count[node_id] = 1

    union_nodes = []
    union_node_tag = []
    for node_id, sum_feat in node_feature_sum.items():
        count = node_feature_count[node_id]
        mean_feat = sum_feat / count  
        union_nodes.append(node_id)
        union_node_tag.append(mean_feat)
        
    return union_nodes, union_node_tag



def RWPE(subgraph):
    #Compute the random walk positional encoding for each node in the subgraph and reduce its dimensionality to the specified dimension. 
    _RW_WALK_LENGTH = 5  
    _RWPE_DIM = 10       
    K = subgraph.shape[0] 
    
    if K == 0:
        return np.array([], dtype=float).reshape(0, _RWPE_DIM) 
    actual_rwpe_dim = min(K, _RWPE_DIM)
    if K == 1: 
        return np.zeros((1, actual_rwpe_dim), dtype=float)
    if not ssp.issparse(subgraph):
        subgraph = ssp.csr_matrix(subgraph)
    degrees = np.asarray(subgraph.sum(axis=1)).flatten() 
    D_inv_vals = np.zeros(K, dtype=float)
    non_zero_degrees_mask = (degrees > 0)
    D_inv_vals[non_zero_degrees_mask] = 1.0 / degrees[non_zero_degrees_mask] 
    D_inv = ssp.diags(D_inv_vals, format='csr')
    P = D_inv @ subgraph
    P_t = P ** _RW_WALK_LENGTH
    P_t_dense = P_t.toarray()
    pca = PCA(n_components=actual_rwpe_dim)
    rwpe_low_dim = pca.fit_transform(P_t_dense)

    return rwpe_low_dim


def edgelist_to_net(csv_path):
    edges = []          
    all_nodes = set()  
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line_num, parts in enumerate(reader, 1):  
            if not parts:  
                continue
            u_raw = parts[0].strip()
            v_raw = parts[1].strip()
            all_nodes.add(u_raw)
            all_nodes.add(v_raw)
            edges.append((u_raw, v_raw, 1))
    
    sorted_nodes = sorted(all_nodes)  
    node_mapping = {raw_id: idx for idx, raw_id in enumerate(sorted_nodes)}
    num_nodes = len(node_mapping)
    
    row = []    
    col = []    
    data = [] 
    
    for u_raw, v_raw, weight in edges:
        u_idx = node_mapping[u_raw]
        v_idx = node_mapping[v_raw]
   
        row.append(u_idx)
        col.append(v_idx)
        data.append(weight)
        
        row.append(v_idx)
        col.append(u_idx)
        data.append(weight)
    
    net = ssp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    
    print(f"total node number: {num_nodes}")
    return net

""" multiclass net"""
def edgelist_txt_to_net(edgelist_path, is_weighted=False):
    """
    从边列表文件读取数据，同时生成网络邻接矩阵（net）和边类别标签字典（labels）
    
    参数:
        edgelist_path: 边列表文件路径，格式要求：
                      - 无权重边：每行 `u v label`（如 "1 2 0"，label为边类别）
                      - 带权重边：每行 `u v weight label`（如 "1 2 0.8 0"）
        is_weighted: 边是否带权重（默认 False，需与边列表格式匹配）
    
    返回:
        net: 网络邻接矩阵（scipy.sparse.csr_matrix，shape=(num_nodes, num_nodes)）
        labels: 边类别标签字典（格式 {(i,j): label, ...}，i<j，i/j为映射后的连续索引）
        node_mapping: 节点ID映射字典（原始ID → 连续索引，便于后续节点对应）
        num_nodes: 网络总节点数
    """
    # 步骤1：读取边列表，收集边信息、节点集合、类别标签
    edges = []          # 存储边的 (u_raw, v_raw, weight)（用于构建net）
    label_records = []  # 存储边的 (u_raw, v_raw, label)（用于构建labels）
    all_nodes = set()   # 收集所有原始节点ID，用于生成映射
    
    with open(edgelist_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):  # line_num用于定位错误行
            line = line.strip()
            if not line:  # 跳过空行
                continue
            
            parts = line.split()  # 分割行数据（支持空格/制表符）
            # 校验行格式（根据是否带权重判断最少元素数）
            min_parts = 3
            if len(parts) < min_parts:
                raise ValueError(
                    f"第{line_num}行格式错误：{line}\n"
                    f"要求格式：{'u v weight label' if is_weighted else 'u v label'}"
                )
            
            # 解析原始节点ID（转为字符串，兼容数字/字符混合ID）
            u_raw = parts[0]
            v_raw = parts[1]
            all_nodes.add(u_raw)
            all_nodes.add(v_raw)
            
            # 解析权重（带权重时）和类别标签
            if is_weighted:
                weight = int(parts[2])
                label = int(parts[2])  # 类别标签（可根据需求转为int/float）
                edges.append((u_raw, v_raw, weight))

            
            # 记录边的原始ID和类别（用于后续生成labels）
            label_records.append((u_raw, v_raw, label))
    
    # 步骤2：生成节点ID映射（原始ID → 0~num_nodes-1的连续索引）
    sorted_nodes = sorted(all_nodes)  # 排序确保映射结果稳定（可选但推荐）
    node_mapping = {raw_id: idx for idx, raw_id in enumerate(sorted_nodes)}
    num_nodes = len(node_mapping)
    
    # 步骤3：构建邻接矩阵net（scipy.sparse.csr_matrix）
    row = []    # 行索引（映射后的节点索引）
    col = []    # 列索引（映射后的节点索引）
    data = []   # 边的权重
    
    for u_raw, v_raw, weight in edges:
        u_idx = node_mapping[u_raw]
        v_idx = node_mapping[v_raw]
        # 无向图：添加双向边（u→v 和 v→u）；若为有向图，删除v→u的三行
        row.append(u_idx)
        col.append(v_idx)
        data.append(weight)
        
        row.append(v_idx)
        col.append(u_idx)
        data.append(weight)
    
    net = ssp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    
    # 步骤4：构建边类别标签字典labels（键为 (i,j) 且 i<j）
    labels = {}
    for u_raw, v_raw, label in label_records:
        u_idx = node_mapping[u_raw]
        v_idx = node_mapping[v_raw]
        # 确保键的格式为 (i,j) 且 i<j（与net的上三角矩阵对应，避免重复）
        if u_idx < v_idx:
            edge_key = (u_idx, v_idx)
        else:
            edge_key = (v_idx, u_idx)
        # 若有重复边（同一u_raw-v_raw出现多次），以最后一行的标签为准
        labels[edge_key] = label
    print("total nodes is:",num_nodes)
    return net, labels


#多分类训练样本划分
import math
import random
import numpy as np
import scipy.sparse as ssp
from collections import defaultdict



def sample_multiclass(net, labels, test_ratio=0.1, train_samples=None, test_samples=None,
                      max_train_num=None, max_test_num=None, all_unknown_as_negative=False, negative_label=0):
    """
    多分类版本的样本划分函数（仅返回正样本，不进行负采样）
    支持：1. 按类别比例划分训练/测试集；2. 用max_train_num限制训练集正样本总量；
         3. 用max_test_num限制测试集正样本总量；4. 确保训练集和测试集内每类别样本均衡
    
    参数:
        net: 输入的网络邻接矩阵（稀疏矩阵，如ssp.csr_matrix）
        labels: 边的类别标签字典，格式为 {(i,j): label, ...}，其中i<j（仅正样本有标签）
        test_ratio: 测试集占总正样本的比例，默认0.1
        train_samples: 预定义的训练正样本，格式为 (节点对列表, 标签列表)，若为None则自动划分
        test_samples: 预定义的测试正样本，格式为 (节点对列表, 标签列表)，若为None则自动划分
        max_train_num: 训练集正样本的最大总量（按类别分布均衡截取）
        max_test_num: 测试集正样本的最大总量（按类别分布均衡截取）
        all_unknown_as_negative: （兼容参数，无实际作用）
        negative_label: （兼容参数，无实际作用）
    
    返回:
        train_data: 训练集数据（仅含正样本，总量≤max_train_num，类别分布均衡）
        test_data: 测试集数据（仅含正样本，总量≤max_test_num，类别分布均衡）
        pos_stats: 正样本划分统计字典，含各类别在训练/测试集的数量分布及均衡性信息
    """
    # 1. 基础处理
    net_triu = ssp.triu(net, k=1)
    n = net.shape[0]
    total_pos = len(labels)
    pos_stats = {
        "total_pos": total_pos,
        "class_dist": defaultdict(dict),
        "max_train_num_used": max_train_num,
        "max_test_num_used": max_test_num
    }

    # 2. 处理正样本划分
    print("------------------------:",  train_samples  )
    if train_samples is None and test_samples is None:
        # 2.1 提取所有正样本及标签并按类别分组
        all_pos_edges = list(labels.keys())
        all_pos_labels = [labels[edge] for edge in all_pos_edges]
        
        class_groups = defaultdict(list)
        class_total = defaultdict(int)
        for idx, (edge, label) in enumerate(zip(all_pos_edges, all_pos_labels)):
            class_groups[label].append((idx, edge))
            class_total[label] += 1
        class_ratio = {label: cnt / total_pos for label, cnt in class_total.items()}
        
        # 2.2 按类别划分初始训练/测试集（未应用数量限制）
        train_pos_edges_by_class = defaultdict(list)
        train_pos_labels_by_class = defaultdict(list)
        test_pos_edges_by_class = defaultdict(list)
        test_pos_labels_by_class = defaultdict(list)
        
        for label, items in class_groups.items():
            class_cnt = len(items)
            sample_indices, edges = zip(*items)
            shuffled_idx = random.sample(range(class_cnt), class_cnt)
            split_idx = int(math.ceil(class_cnt * (1 - test_ratio)))
            
            # 训练集（初始）
            train_idx = [sample_indices[i] for i in shuffled_idx[:split_idx]]
            train_edges = [all_pos_edges[idx] for idx in train_idx]
            train_labels = [all_pos_labels[idx] for idx in train_idx]
            train_pos_edges_by_class[label] = train_edges
            train_pos_labels_by_class[label] = train_labels
            
            # 测试集（初始）
            test_idx = [sample_indices[i] for i in shuffled_idx[split_idx:]]
            test_edges = [all_pos_edges[idx] for idx in test_idx]
            test_labels = [all_pos_labels[idx] for idx in test_idx]
            test_pos_edges_by_class[label] = test_edges
            test_pos_labels_by_class[label] = test_labels
            
            # 记录初始统计
            pos_stats["class_dist"][label]["pre_train_count"] = len(train_edges)
            pos_stats["class_dist"][label]["pre_test_count"] = len(test_edges)
            pos_stats["class_dist"][label]["total_count"] = class_cnt
            pos_stats["class_dist"][label]["class_ratio"] = class_ratio[label]
        
        # 2.3 应用max_train_num：按类别比例均衡截取训练集
        train_pos_edges, train_pos_labels = [], []
        if max_train_num is not None and max_train_num > 0:
            class_train_quota = {
                label: int(math.ceil(class_ratio[label] * max_train_num))
                for label in class_groups.keys()
            }
            for label in class_groups.keys():
                actual_take = min(class_train_quota[label], len(train_pos_edges_by_class[label]))
                train_pos_edges.extend(train_pos_edges_by_class[label][:actual_take])
                train_pos_labels.extend(train_pos_labels_by_class[label][:actual_take])
                pos_stats["class_dist"][label]["train_count"] = actual_take
        else:
            for label in class_groups.keys():
                train_pos_edges.extend(train_pos_edges_by_class[label])
                train_pos_labels.extend(train_pos_labels_by_class[label])
                pos_stats["class_dist"][label]["train_count"] = len(train_pos_edges_by_class[label])
        
        # 2.4 应用max_test_num：按类别比例均衡截取测试集
        test_pos_edges, test_pos_labels = [], []
        if max_test_num is not None and max_test_num > 0:
            class_test_quota = {
                label: int(math.ceil(class_ratio[label] * max_test_num))
                for label in class_groups.keys()
            }
            for label in class_groups.keys():
                actual_take = min(class_test_quota[label], len(test_pos_edges_by_class[label]))
                test_pos_edges.extend(test_pos_edges_by_class[label][:actual_take])
                test_pos_labels.extend(test_pos_labels_by_class[label][:actual_take])
                pos_stats["class_dist"][label]["test_count"] = actual_take
        else:
            for label in class_groups.keys():
                test_pos_edges.extend(test_pos_edges_by_class[label])
                test_pos_labels.extend(test_pos_labels_by_class[label])
                pos_stats["class_dist"][label]["test_count"] = len(test_pos_edges_by_class[label])
    
    else:
        # 处理预定义样本的情况
        train_pos_edges, train_pos_labels = train_samples
        test_pos_edges, test_pos_labels = test_samples
        
        # 处理训练集限制
        if max_train_num is not None and max_train_num > 0 and len(train_pos_edges) > max_train_num:
            train_class_count = defaultdict(int)
            for label in train_pos_labels:
                train_class_count[label] += 1
            train_total_pre = len(train_pos_edges)
            class_ratio_train = {label: cnt / train_total_pre for label, cnt in train_class_count.items()}
            
            train_edges_by_class = defaultdict(list)
            train_labels_by_class = defaultdict(list)
            for edge, label in zip(train_pos_edges, train_pos_labels):
                train_edges_by_class[label].append(edge)
                train_labels_by_class[label].append(label)
            
            train_pos_edges, train_pos_labels = [], []
            for label in train_edges_by_class.keys():
                class_quota = int(math.ceil(class_ratio_train[label] * max_train_num))
                actual_take = min(class_quota, len(train_edges_by_class[label]))
                train_pos_edges.extend(train_edges_by_class[label][:actual_take])
                train_pos_labels.extend(train_labels_by_class[label][:actual_take])
                train_class_count[label] = actual_take
        
        # 处理测试集限制
        if max_test_num is not None and max_test_num > 0 and len(test_pos_edges) > max_test_num:
            test_class_count = defaultdict(int)
            for label in test_pos_labels:
                test_class_count[label] += 1
            test_total_pre = len(test_pos_edges)
            class_ratio_test = {label: cnt / test_total_pre for label, cnt in test_class_count.items()}
            
            test_edges_by_class = defaultdict(list)
            test_labels_by_class = defaultdict(list)
            for edge, label in zip(test_pos_edges, test_pos_labels):
                test_edges_by_class[label].append(edge)
                test_labels_by_class[label].append(label)
            
            test_pos_edges, test_pos_labels = [], []
            for label in test_edges_by_class.keys():
                class_quota = int(math.ceil(class_ratio_test[label] * max_test_num))
                actual_take = min(class_quota, len(test_edges_by_class[label]))
                test_pos_edges.extend(test_edges_by_class[label][:actual_take])
                test_pos_labels.extend(test_labels_by_class[label][:actual_take])
                test_class_count[label] = actual_take
        
        # 更新统计信息
        train_class_count = defaultdict(int)
        for label in train_pos_labels:
            train_class_count[label] += 1
        test_class_count = defaultdict(int)
        for label in test_pos_labels:
            test_class_count[label] += 1
        
        total_class = set(train_class_count.keys()).union(test_class_count.keys())
        for label in total_class:
            class_total = train_class_count.get(label, 0) + test_class_count.get(label, 0)
            pos_stats["class_dist"][label] = {
                "train_count": train_class_count.get(label, 0),
                "test_count": test_class_count.get(label, 0),
                "total_count": class_total,
                "class_ratio": class_total / total_pos if total_pos > 0 else 0.0
            }

    # 3. 补充最终统计信息
    final_train_total = len(train_pos_edges)
    final_test_total = len(test_pos_edges)
    
    pos_stats["train_pos_count"] = final_train_total
    pos_stats["test_pos_count"] = final_test_total
    pos_stats["train_total_count"] = final_train_total
    pos_stats["test_total_count"] = final_test_total
    
    # 计算训练集和测试集内的类别占比（验证均衡性）
    for label in pos_stats["class_dist"].keys():
        pos_stats["class_dist"][label]["train_ratio_in_train"] = (
            pos_stats["class_dist"][label]["train_count"] / final_train_total 
            if final_train_total > 0 else 0.0
        )
        pos_stats["class_dist"][label]["test_ratio_in_test"] = (
            pos_stats["class_dist"][label]["test_count"] / final_test_total 
            if final_test_total > 0 else 0.0
        )
    print("final_train_total:",final_train_total  )
    print("final_test_total:",final_test_total  )
    # 4. 准备返回数据
    train_data = (train_pos_edges, train_pos_labels)
    test_data = (test_pos_edges, test_pos_labels)
    return train_data, test_data







# 多分类 links2subgraphs  ； helper可复用。  
def links2subgraphs_ML(A, train_pos,train_lb, test_pos, test_lb, h=3, 
                    max_nodes_per_hop=None, node_information=None, PPR_arrays=None, no_parallel=False,model_name="M-SEAL",multiplexing_count=1):

    # extract enclosing subgraphs
    def helper(A, links, g_label):
        iqo=0
        g_list = []
        link_data = list(zip(links[0], links[1]))

        print("--------------",model_name  )
        if no_parallel:
            count_meg=0   #  MIMO MODEL
            meg_list = []
            step_len = multiplexing_count
            for step_idx in range(0,len(link_data),step_len): 
                if model_name == "M-SEAL" or model_name == "M-PS2":
                    g, n_labels, n_features, subgraph2list,len_out,subgraph_RPE = subgraph_extraction_SEAL(
                        list(link_data[step_idx:step_idx+step_len]), A, h, max_nodes_per_hop, node_information,PPR_arrays
                    )
                elif model_name == "M-SMA":
                    g, n_labels, n_features, subgraph2list,len_out,subgraph_RPE = subgraph_extraction_SMA(
                        list(link_data[step_idx:step_idx+step_len]), A, h, max_nodes_per_hop, node_information,PPR_arrays
                    )                
                elif model_name == "M-LPFormer":
                    g, n_labels, n_features, subgraph2list,len_out,subgraph_RPE = subgraph_extraction_LPFormer(
                        list(link_data[step_idx:step_idx+step_len]), A, h, max_nodes_per_hop, node_information,PPR_arrays
                    )
                elif model_name == "M-NCN":
                    g, n_labels, n_features, subgraph2list,len_out,subgraph_RPE = subgraph_extraction_NCN(
                        list(link_data[step_idx:step_idx+step_len]), A, h, max_nodes_per_hop, node_information,PPR_arrays
                    )
                
                g_list.append(GNNGraph(g, list(g_label[step_idx:step_idx+step_len]), n_labels, n_features,subgraph2list,len_out,subgraph_RPE))
                
            return g_list

            
            # step_len = multiplexing_count
            # for step_idx in range(0,len(link_data),step_len):
            #     #count_meg= count_meg+1
            #     #meg_list.append( (i, j) )
            #     #if count_meg == 6000:
            #         #count_meg=0 
            #     g, n_labels, n_features, subgraph2list,len_out = subgraph_extraction_labeling(
            #         list(link_data[step_idx:step_idx+step_len]), A, h, max_nodes_per_hop, node_information
            #     )
            #     #max_n_label['value'] = max(max(n_labels), max_n_label['value'])

            #     g_list.append(GNNGraph(g, list(g_label[step_idx:step_idx+step_len]), n_labels, n_features,subgraph2list,len_out))
                    #g_list.append(GNNGraph(g, g_label[iqo:iqo+5000], n_labels, n_features,subgraph2list,len_out))
                    #iqo = iqo+5000
                    #meg_list = []
                #elif len(links[0]) - count_meg < 5000:
                    

            
            #return g_list
        else:
            # the parallel extraction code
            start = time.time()
            pool = mp.Pool(mp.cpu_count())
            results = pool.map_async(
                parallel_worker, 
                [((i, j), A, h, max_nodes_per_hop, node_information) for i, j in zip(links[0], links[1])]
            )
            remaining = results._number_left
            pbar = tqdm(total=remaining)
            while True:
                pbar.update(remaining - results._number_left)
                if results.ready(): break
                remaining = results._number_left
                time.sleep(1)
            results = results.get()
            pool.close()
            pbar.close()
            g_list = [GNNGraph(g, g_label, n_labels, n_features) for g, n_labels, n_features in results]
            #max_n_label['value'] = max(
            #    max([max(n_labels) for _, n_labels, _ in results]), max_n_label['value']
            #)
            end = time.time()
            print("Time eplased for subgraph extraction: {}s".format(end-start))
            return g_list
    #clusIdx = np.loadtxt('./clusterIdx.txt', dtype=int,  delimiter=',')
    print('Enclosing subgraph extraction begins...')
    
    train_graphs, test_graphs = None, None
    pos_per_batch= 1
    neg_per_batch= 1    
    if train_pos:
        
        train_ind0 = np.array(train_pos[0])
        train_ind1 = np.array(train_pos[1])
        train_lb = np.array(train_lb)
        
        help_ind = np.arange(len(train_ind0))
        np.random.seed(4234)
        np.random.shuffle(help_ind)    
        train_ind0 = train_ind0[help_ind]
        train_ind1 = train_ind1[help_ind]
        train_lb = train_lb[help_ind]
        
        train_sub_matrix = []
        for i,j in zip(train_ind0, train_ind1):
            subgraph_feature = subgraph_embedding((i,j), A, h, node_information ,max_nodes_per_hop)
            train_sub_matrix.append(subgraph_feature)     
        cluster_labels = cluster_subgraph_features(np.array(train_sub_matrix),1)
        df = pd.DataFrame(cluster_labels)
        df_sorted = df.sort_values(by=0)
        help_ind = list(df_sorted[0].index)
        print("start train_graph extract!!!!!")
        train_graphs = helper(A, (train_ind0[help_ind], train_ind1[help_ind] ), train_lb[help_ind] )  
        print("end train_graph extract!!!!!")
    if test_pos:
        test_ind0 = np.array(test_pos[0])
        test_ind1 = np.array(test_pos[1])
        test_lb = np.array(test_lb)
       
        help_ind = np.arange(len(test_ind0))
        np.random.shuffle(help_ind)    
        test_ind0 = test_ind0[help_ind]
        test_ind1 = test_ind1[help_ind]
        test_lb = test_lb[help_ind]        
        
        test_sub_matrix = []
        for i,j in zip(test_ind0, test_ind1):
            subgraph_feature = subgraph_embedding((i,j), A, h, node_information ,max_nodes_per_hop)
            test_sub_matrix.append(subgraph_feature)
        cluster_labels = cluster_subgraph_features(np.array(test_sub_matrix),1)        
        df = pd.DataFrame(cluster_labels)
        df_sorted = df.sort_values(by=0)        
        help_ind = list(df_sorted[0].index)
        print("start test_graph extract!!!!!")
        test_graphs = helper(A, (test_ind0[help_ind],test_ind1[help_ind]), test_lb[help_ind])  
        print("end test_graph extract!!!!!")
    return train_graphs, test_graphs




def subgraph_extraction_labeling(inds, A, h=1, max_nodes_per_hop=None,
                                 node_information=None):
    # extract the h-hop enclosing subgraph around link 'ind'   
    #concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
    #node_tag = torch.zeros(n_nodes, 32)
    #node_tag.scatter_(1, concat_tag, 1)    
    concat_tag=[]
    concat_nodes=[]
    
    #graph_idx = 0
    #step_len = 10
    #graph_demux_feat = np.zeros((step_len,step_len),dtype=int)
    #graph_demux_feat[np.arange(step_len), np.arange(step_len) ]=1
    
    for ind in inds:
        dist = 0
        nodes = set( [])
        visited = set([])
        fringe = set([])
        nodes = nodes.union(set([ind[0], ind[1]])) 
        visited = visited.union(  set([ind[0], ind[1]]))
        fringe = fringe.union(set([ind[0], ind[1]]))    
        for dist in range(1, h+1):
            fringe = neighbors(fringe, A)
            fringe = fringe - visited
            visited = visited.union(fringe)
            if max_nodes_per_hop is not None:
                if max_nodes_per_hop < len(fringe):
                    fringe = random.sample(fringe, max_nodes_per_hop)
            if len(fringe) == 0:
                break
            nodes = nodes.union(fringe)
        if ind[0] in nodes:
            nodes.remove(ind[0])
        if ind[1] in nodes:
            nodes.remove(ind[1])
        nodes = list([ind[0],ind[1]]) + list(nodes)
        subgraph = A[nodes, :][:, nodes]       # obtain subgraph -------- important  checkpoints -------------------
        #print("single subgraph nodes:",nodes)
        #node_tag = RWPE(subgraph)
        #print("!!:", node_tag)
        labels = node_label(subgraph)
    
        #node_tag = hash_encode( labels , 50 )
        
        node_tag = np.zeros((len(nodes),100),dtype=int)  # 32 one-hot
        
        node_tag[np.arange(len(nodes)), labels]=1
        
        #graph_feat = graph_demux_feat[graph_idx]  # 获取对应图的特征
        # 确保graph_feat是二维的，如果是一维则添加一个维度
        #if graph_feat.ndim == 1:
        #    graph_feat = graph_feat.reshape(1, -1)
        # 重复特征，使其与节点数量匹配，保持二维结构
        #graph_demux_feat_h = np.repeat(graph_feat, len(nodes), axis=0)

        # 确认两个数组都是二维的再拼接
        #if graph_demux_feat_h.ndim == 2 and node_tag.ndim == 2:
        #    node_tag = np.concatenate([graph_demux_feat_h, node_tag], axis=1)        
        #else:
        #    print("error!!!!!!")
        
        
        #graph_idx = graph_idx+1
        concat_tag.append(node_tag)
        concat_nodes.append(nodes)
        
    u_nodes,u_node_tags = sum_same_node_features(concat_nodes,concat_tag)    
         
    pnodes = set([])
    for ind in inds:
        pnodes = pnodes.union(set([ind[0], ind[1]])) 
    
    len_out = len(pnodes)
    subgraph2list = []
    for ind in inds:
        ind_posA = list(pnodes).index(ind[0])
        ind_posB = list(pnodes).index(ind[1])
        subgraph2list.append( (ind_posA,ind_posB) )      
     
    """
    temp_embs=[]
    for main_node in pnodes:    
        if main_node in u_nodes:
            ind_pos = list(u_nodes).index(main_node)
            u_nodes.remove(main_node)
            temp_embs.append(u_node_tags[ind_pos])
            del u_node_tags[ind_pos]
        
    u_nodes = list(pnodes) + list(u_nodes) 
    u_node_tags = temp_embs + u_node_tags
    """
    #print("union graph nodes:" , u_nodes )
    #print("union graph labels:" , u_node_tags )
    #print("the number fo nodes:",nodes)
    subgraph = A[u_nodes, :][:, u_nodes]
    #print("subgraph:",subgraph)
    # apply node-labeling
    #labels = node_label(subgraph)
    #print("labels:",labels)
    # get node features
    features = None
    if node_information is not None:
        features = node_information[u_nodes]
    # construct nx graph
    g = nx.from_numpy_array(subgraph)
    # remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)
    

    return g, u_node_tags, features, subgraph2list, len_out



# def sum_same_node_features(subgraphs, features):
#     # 找到最大的子图（节点数量最多）作为基准
#     max_size = max(len(subgraph) for subgraph in subgraphs) if subgraphs else 0
#     max_subgraph = next((sg for sg in subgraphs if len(sg) == max_size), [])
    
#     # 初始化节点特征累加字典和计数字典（使用最大子图的节点ID作为统一标识）
#     node_feature_sum = {node_id: np.zeros_like(features[0][0]) for node_id in max_subgraph}
#     node_feature_count = {node_id: 0 for node_id in max_subgraph}
    
#     # 遍历每个子图（作为一个"句子"）及其特征
#     for subgraph, feat_list in zip(subgraphs, features):
#         # 填充子图到最大长度，使用最大子图的节点ID进行填充
#         padded_subgraph = []
#         padded_feats = []
        
#         # 添加原始子图的节点和特征
#         for node_id, feat in zip(subgraph, feat_list):
#             padded_subgraph.append(node_id)
#             padded_feats.append(feat)
        
#         # 填充剩余位置，使用最大子图对应位置的节点ID和零向量
#         for i in range(len(subgraph), max_size):
#             # 使用最大子图中对应位置的节点ID
#             pad_node_id = max_subgraph[i] if i < len(max_subgraph) else f"pad_{i}"
#             padded_subgraph.append(pad_node_id)
#             padded_feats.append(np.zeros_like(features[0][0]))
        
#         # 按照位置顺序累加特征，使用最大子图的节点ID作为统一标识
#         for pos in range(max_size):
#             # 始终使用最大子图中该位置的节点ID作为键
#             key = max_subgraph[pos]
#             # 累加对应位置的特征
#             node_feature_sum[key] += np.array(padded_feats[pos])
#             node_feature_count[key] += 1
    
#     # 计算平均值
#     union_nodes = []
#     union_node_tag = []
#     # 按照最大子图的节点顺序返回结果
#     for node_id in max_subgraph:
#         sum_feat = node_feature_sum[node_id]
#         count = node_feature_count[node_id]
#         mean_feat = sum_feat / count if count > 0 else sum_feat  # 避免除以零
#         union_nodes.append(node_id)
#         union_node_tag.append(mean_feat)
        
        
#     return union_nodes, union_node_tag



''' subgraph multiplexing and union graph construction  ''' 
def sum_same_node_features(subgraphs, features):
    
    node_feature_sum = {}
    node_feature_count = {}

    for subgraph, feat_list in zip(subgraphs, features):
        
        for node_id, feat in zip(subgraph, feat_list):
            if node_id in node_feature_sum:
                
                node_feature_sum[node_id] = np.array([
                    a + b for a, b in zip(node_feature_sum[node_id], feat)
                ])
                node_feature_count[node_id] += 1
            else:
                node_feature_sum[node_id] = feat
                node_feature_count[node_id] = 1

    union_nodes = []
    union_node_tag = []
    for node_id, sum_feat in node_feature_sum.items():
        count = node_feature_count[node_id]
        mean_feat = sum_feat / count  
        union_nodes.append(node_id)
        union_node_tag.append(mean_feat)
        
    return union_nodes, union_node_tag


