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
from torch.nn import MarginRankingLoss

sys.path.append('%s/lib' % os.path.dirname(os.path.realpath(__file__)))
from pytorch_util import weights_init




''' classifier using Cross-Entropy loss function + 最后一层正交正则 '''
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, with_dropout,loss_beta):
        super(MLPClassifier, self).__init__()
        self.h1_weights = nn.Linear(input_size, hidden_size)  # input_size
        self.h2_weights = nn.Linear(hidden_size, num_class)  # 最后一层
        self.with_dropout = with_dropout
        self.beta = loss_beta
        weights_init(self)

    def orthogonal_reg_last_layer(self, lambda_reg=1e-4):
        """
        只对 最后一层权重 h2_weights 做正交正则
        """
        W = self.h2_weights.weight  # 只取最后一层
        WtW = torch.matmul(W, W.T)   # 形状 [num_class, num_class]
        I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
        reg_loss = lambda_reg * torch.norm(WtW - I, p='fro') ** 2
        return reg_loss

    def forward(self, x, y = None, epoch = None, lambda_reg=1e-4):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)        
        logits = self.h2_weights(h1)
        
        # 原代码重复 softmax 已删除，只保留 log_softmax 即可
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            ce_loss = F.nll_loss(logits, y)
            
            # ✅ 只加最后一层的正交正则
            reg_loss = self.orthogonal_reg_last_layer(lambda_reg=lambda_reg)
            total_loss = ce_loss + self.beta * reg_loss
            
            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            return logits, total_loss, acc
        else:
            return logits
        
