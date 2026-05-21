import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
from model_SEAL import SEAL
from model_SMA import SMA
from model_LPFormer import LPFormer
from model_NCN import NCN
from mlp_dropout import MLPClassifier
from sklearn import metrics
from util import cmd_args, load_data
import time
from itertools import chain  

class Classifier(nn.Module):
    def __init__(self, regression=False):
        super(Classifier, self).__init__()
        self.regression = regression
        self.aggeTime = 0
        self.comput_num = 0
        if cmd_args.gm == 'M-SEAL' or cmd_args.gm == 'M-PS2':
            model = SEAL
        elif cmd_args.gm == 'M-NCN':
            model = NCN
        elif cmd_args.gm == 'M-SMA':
            model = SMA            
        elif cmd_args.gm == 'M-LPFormer':
            model = LPFormer

        self.gnn = model(latent_dim=cmd_args.latent_dim,
                        output_dim=cmd_args.out_dim,
                        num_node_feats=cmd_args.feat_dim+cmd_args.attr_dim,
                        num_edge_feats=cmd_args.edge_feat_dim,
                        conv1d_activation=cmd_args.conv1d_activation,
                        multiplexing_count=cmd_args.multiplexing_count                        
                        )
     
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            out_dim = self.gnn.dense_dim
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.hidden, num_class=cmd_args.num_class, with_dropout=cmd_args.dropout, loss_beta = cmd_args.beta )

        
    def PrepareFeatureLabel(self, batch_graph):

        labels = []
        n_nodes = 0
    
        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
        else:
            node_tag_flag = False
        
        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False
            
        if cmd_args.edge_feat_dim > 0:
            edge_feat_flag = True
            concat_edge_feat = []
        else:
            edge_feat_flag = False
        
        for i in range(len(batch_graph)):    
            labels = np.concatenate((labels,batch_graph[i].label )   )
            n_nodes += batch_graph[i].num_nodes

            if node_feat_flag == True:
                tmp = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
                concat_feat.append(tmp)
            if edge_feat_flag == True:
                if batch_graph[i].edge_features is not None:  # in case no edge in graph[i]
                    tmp = torch.from_numpy(batch_graph[i].edge_features).type('torch.FloatTensor')
                    concat_edge_feat.append(tmp)
        labels = torch.LongTensor(labels)
        if node_tag_flag == True:
            node_tag = torch.tensor( batch_graph[0].node_tags, dtype=torch.float32)
        
        if node_feat_flag == True:
            node_feat = torch.cat(concat_feat, 0)
        
        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features
        
        if edge_feat_flag == True:
            edge_feat = torch.cat(concat_edge_feat, 0)

        if cmd_args.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()
            if edge_feat_flag == True:
                edge_feat = edge_feat.cuda()

        if edge_feat_flag == True:
            return node_feat, edge_feat, labels
        return node_feat, labels

    def forward(self, batch_graph,epoch):
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        time_st1 = time.time()    
        embed = self.gnn(batch_graph, node_feat, edge_feat,epoch)
        time_ed1 = time.time()
        self.totalnum1 = self.gnn.totalnum1
        self.comput_num = self.gnn.comput_num
        return self.mlp(embed, labels,epoch)

    def output_features(self, batch_graph):
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat)
        return embed, labels
        

# HR@K
def calculate_hit_for_batch(targets, scores, k=50):
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i][1], reverse=True)
    topk_indices = sorted_indices[:k]
    has_relevant = any(targets[i] == 1 for i in topk_indices)
    return 1 if has_relevant else 0


def loop_dataset(g_list, classifier, sample_idxes, optimizer=None, bsize=cmd_args.batch_size, totTime=None, epoch = None):
    
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters),unit='batch',position=0 ,leave=True)
    all_targets = []
    all_scores = []
    batch_targets = [] 
    batch_scores = []  
    hit_results = []    

    n_samples = 0
    time_st = time.time()
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]
        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]

        all_targets = list(chain(all_targets, targets[0]))
        logits, loss, acc = classifier(batch_graph,epoch)
        
        if cmd_args.dn =="Collab":        
            batch_targets.append(targets[0][0])  
            batch_scores.append(logits.detach().cpu().numpy()[0])  
            
            if len(batch_targets) >= 50:
                current_hit = calculate_hit_for_batch(batch_targets, batch_scores, k=50)
                hit_results.append({
                    'hit_at_50': current_hit
                })
               
                batch_targets = []
                batch_scores = []
        elif cmd_args.dn =="PPA": 
            batch_targets.append(targets[0][0])  
            batch_scores.append(logits.detach().cpu().numpy()[0])  
            
            if len(batch_targets) >= 100:
                current_hit = calculate_hit_for_batch(batch_targets, batch_scores, k=100)
                hit_results.append({
                    'hit_at_50': current_hit
                })
               
                batch_targets = []
                batch_scores = []

        all_scores.append(logits[:, 1].cpu().detach())
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().detach().numpy()
        total_loss.append( np.array([loss, acc]) * len(selected_idx) )
        n_samples += len(selected_idx)
    time_end = time.time()
    if totTime is not None:
        totTime+= (time_end-time_st)

    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()
    
    # cal hit@
    if hit_results:
        overall_hit = sum(item['hit_at_50'] for item in hit_results) / len(hit_results)
    else:
        overall_hit = 0

    if overall_hit == 0:
        fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        avg_loss = np.concatenate((avg_loss, [auc]))
    else:
        avg_loss = np.concatenate((avg_loss, [overall_hit]))

    return avg_loss,totTime

