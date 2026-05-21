import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
from torch.utils.data import DataLoader
sys.path.append('%s/Models' % os.path.dirname(os.path.realpath(__file__)))
from classifier import *
from util_functions import *
torch.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser(description='Link Prediction with MIMO-LP')
# general settings
parser.add_argument('--model-name', default='M-SEAL', choices=["M-SEAL","M-PS2","M-SMA","M-NCN","M-LPFormer"], help='select SB backbone')
parser.add_argument('--data-name', default='NS', choices=["Drugbank","Yeast","NS","Friendster","Collab","PPA","Facebook","com-Orkut","WikiKG90Mv2","Random graph","4-Regular","Musae-chameleon","Ecoli","Wikipedia"], help='select datasets')
parser.add_argument('--multiplexing-count',type=int , default=1, help='multiplexing count')
parser.add_argument('--hop', default=2, metavar='S',  
                    choices=["2","3","4"],
                    help='subgraph hop number'  )
parser.add_argument('--max-train-num', type=int, default=100000, 
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed')
parser.add_argument('--test-ratio', type=float, default=0.2,
                    help='ratio of test links')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)
torch.manual_seed(cmd_args.seed)
args.hop = int(args.hop)



'''Step1-2. Subgraph Samples Extraction and Featrues Assignment'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))

# check whether train and test links are provided
train_pos, test_pos = None, None
# build observed network
APPENDING_DATA_NAMES = ["PPA", "Collab"]
MULTI_CLASS_DATASETS = ["Drugbank"]
if args.data_name is not None:  # use .mat network

    if args.data_name in APPENDING_DATA_NAMES:
        net = edgelist_to_net( os.path.join("data", f"{args.data_name}.csv"))
    elif args.data_name in MULTI_CLASS_DATASETS:
        net, labels = edgelist_txt_to_net(os.path.join("data", f"{args.data_name}.txt"), is_weighted=True )
    else:
        args.data_dir = os.path.join(args.file_dir, 'data/{}.mat'.format(args.data_name))    
        data = sio.loadmat(args.data_dir)
        net = data['net']
        net = net.tocsr()
else:  # build network from train links
    print("lacking datasets")

# sample train and test links
if args.data_name in APPENDING_DATA_NAMES:
    train_pos, train_neg, test_pos, test_neg = sample_neg_in_HR(
        net, args.test_ratio, max_train_num=args.max_train_num
    )
elif args.data_name in MULTI_CLASS_DATASETS:
    train_data, test_data = sample_multiclass(
    net, labels, args.test_ratio
    )
else:
    train_pos, train_neg, test_pos, test_neg = sample_neg_in_AUC(
        net, args.test_ratio, max_train_num=args.max_train_num
    )


if args.data_name in MULTI_CLASS_DATASETS:
    train_edges, train_lb = train_data  # 拆分节点对和标签
    train_u = [edge[0] for edge in train_edges]  # 所有边的起点
    train_v = [edge[1] for edge in train_edges]  # 所有边的终点
    train_pos = (train_u, train_v)   # 转换为 (u_list, v_list) 格式    
        
    # 转换测试数据
    test_edges, test_lb = test_data
    test_u = [edge[0] for edge in test_edges]
    test_v = [edge[1] for edge in test_edges]
    test_pos = (test_u, test_v)


    A = net.copy()  # the observed network
    #ssp.save_npz('observed_network.npz', A)  # save A for clustering
    
    A[test_pos[0], test_pos[1]] = 0  # mask test links
    A[test_pos[1], test_pos[0]] = 0  # mask test links
    A.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x
    
    node_information = None
    embeddings = generate_node2vec_embeddings(A, 128, False, None)
    PPR_arrays = compute_PPR_viaSim(embeddings)
    node_information = embeddings
    if args.use_attribute and attributes is not None:
        if node_information is not None:
            node_information = np.concatenate([node_information, attributes], axis=1)
        else:
            node_information = attributes
    print("start~~~!!!!!!")        
    train_graphs, test_graphs = links2subgraphs_ML(A, train_pos,train_lb, test_pos, test_lb, args.hop,  node_information,PPR_arrays,args.model_name, args.multiplexing_count)      
    print("end~~~!!!!!!")    
else:
    A = net.copy()  # the observed network
    A[test_pos[0], test_pos[1]] = 0  # mask test links
    A[test_pos[1], test_pos[0]] = 0  # mask test links
    A.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x
    
    node_information = None
    embeddings = generate_node2vec_embeddings(A, 128, True, train_neg)
    PPR_arrays = compute_PPR_viaSim(embeddings)
    node_information = embeddings
    
    train_graphs, test_graphs = links2subgraphs(
        A, 
        train_pos, 
        train_neg, 
        test_pos, 
        test_neg, 
        args.hop, 
        node_information, 
        PPR_arrays,
        args.model_name,
        args.multiplexing_count
    )


'''Step3. Node Representation and Prediction'''   
cmd_args.gm = args.model_name
cmd_args.dn = args.data_name
cmd_args.multiplexing_count = args.multiplexing_count
cmd_args.latent_dim = [256,256]
cmd_args.hidden = 256
cmd_args.out_dim = 0
cmd_args.dropout = True
if cmd_args.dn == "Drugbank":
    cmd_args.num_class = 86
else:
    cmd_args.num_class = 2
cmd_args.mode = 'gpu' if args.cuda else 'cpu'
cmd_args.num_epochs =100
cmd_args.learning_rate = 1e-4
cmd_args.printAUC = True
cmd_args.feat_dim = 100
cmd_args.attr_dim = 0
cmd_args.beta = 0
if node_information is not None:
    cmd_args.attr_dim = node_information.shape[1]

classifier = Classifier()
if cmd_args.mode == 'gpu':
    classifier = classifier.cuda()

optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate,weight_decay=0.001)

train_idxes = list(range(len(train_graphs)))
best_loss = None
best_epoch = None
time_start = time.time() 

totTime=0
# ===================== 初始化记录 =====================
best_auc = -1.0          # 记录最佳测试AUC
best_auc_epoch = 0      # 最佳AUC所在轮次
best_auc_loss = 0.0     # 最佳AUC对应的损失

train_auc_history = []  # 保存每一轮训练AUC
test_auc_history = []   # 保存每一轮测试AUC

for epoch in range(cmd_args.num_epochs):
    classifier.train()
    avg_loss,updateTime = loop_dataset(
        train_graphs, classifier, train_idxes, optimizer=optimizer, bsize=1,totTime = totTime,epoch=epoch
    )
    totTime = updateTime
    
    train_auc = avg_loss[2]
    train_auc_history.append(train_auc)  # 收集训练AUC
    print('\033[92maverage training of epoch %d: loss %.5f auc %.5f\033[0m' % (
        epoch, avg_loss[0], train_auc))

    classifier.eval()
    test_loss,updateTime = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))) , bsize=1,totTime = totTime,epoch=epoch)
    totTime = updateTime

    test_auc = test_loss[2]
    test_auc_history.append(test_auc)    # 收集测试AUC

    # ===================== ✅ 核心：按 AUC 选择最佳模型 =====================
    if test_auc > best_auc:
        best_auc = test_auc
        best_auc_epoch = epoch
        best_auc_loss = test_loss[0]

    print('\033[94maverage test of epoch %d: loss %.5f  auc %.5f\033[0m' % (
            epoch, test_loss[0],  test_auc))

time_end = time.time()
print("---------aggreTime:",classifier.aggeTime)
print("---------n2nlength:",classifier.totalnum1)
print("time cost is:",time_end-time_start)

# ===================== ✅ 输出最终最佳 AUC =====================
print('\033[95m===============================================\033[0m')
print('\033[95m  Best Test AUC : epoch %d: loss %.5f  auc %.5f\033[0m' % (
    best_auc_epoch, best_auc_loss, best_auc))
print('\033[95m===============================================\033[0m')

print("vol analysing")    
print(classifier.totalnum1 )   
print(classifier.comput_num )   
print("average value:",classifier.totalnum1/ classifier.comput_num   )  
