import numpy as np
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from torch import FloatTensor
from torch.nn.parameter import Parameter
from layers import GraphConvolution, SimilarityAdj, DistanceAdj
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.1)

def weight_init2(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        torch_init.xavier_normal_(m.weight.data)
        torch_init.constant_(m.bias.data, 0.01)
    elif classname.find('Linear') != -1:
        torch_init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch_init.constant_(m.bias.data, 0.01)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        n_features = args.feature_size
        n_class = args.num_classes

        self.conv1d1 = nn.Conv1d(in_channels=n_features, out_channels=512, kernel_size=1, padding=0)
        self.conv1d2 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1, padding=0)
        self.conv1d3 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=5, padding=2)
        self.conv1d4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        # Graph Convolution
        self.gc1 = GraphConvolution(128, 32, residual=True)  # nn.Linear(128, 32)
        self.gc2 = GraphConvolution(32, 32, residual=True)
        self.gc3 = GraphConvolution(128, 32, residual=True)  # nn.Linear(128, 32)
        self.gc4 = GraphConvolution(32, 32, residual=True)
        self.gc5 = GraphConvolution(128, 32, residual=True)  # nn.Linear(128, 32)
        self.gc6 = GraphConvolution(32, 32, residual=True)
        self.simAdj = SimilarityAdj(n_features, 32)
        self.disAdj = DistanceAdj()

        # self.classifier = nn.Linear(32*3, n_class)  ###2019-11-12
        self.classifier = nn.Linear(32*3, n_class, bias=False)
        # self.classifierMid = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1, bias=False))
        # self.classifierMid = nn.Sequential(nn.Conv1d(128, 64, 5, padding=2), nn.ReLU(), nn.Conv1d(64, 32, 5, padding=2),
        #                                    nn.ReLU(), nn.Conv1d(32, 1, 5, padding=2))
        self.approximator = nn.Sequential(nn.Conv1d(128, 64, 1, padding=0), nn.ReLU(),
                                           nn.Conv1d(64, 32, 1, padding=0), nn.ReLU())
        self.conv1d_approximator = nn.Conv1d(32, 1, 5, padding=0)
        self.dropout = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)



    def forward(self, inputs, seq_len):
        x = inputs.permute(0, 2, 1)  # for conv1d
        x = self.relu(self.conv1d1(x))
        x = self.dropout(x)
        x = self.relu(self.conv1d2(x))
        x = self.dropout(x)

        logits = self.approximator(x)
        logits = F.pad(logits, (4, 0))
        logits = self.conv1d_approximator(logits)
        logits = logits.permute(0,2,1)
        x = x.permute(0, 2, 1)  # b*t*c
        # x = x.repeat(1,1,3)
        x = x[:, :, :96]
        # print(x.shape)
        x = self.classifier(x)
        return x, logits


