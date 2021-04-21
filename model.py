
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from layers import GraphConvolution, SimilarityAdj, DistanceAdj


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.1)

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

        self.classifier = nn.Linear(32*3, n_class)
        self.approximator = nn.Sequential(nn.Conv1d(128, 64, 1, padding=0), nn.ReLU(),
                                          nn.Conv1d(64, 32, 1, padding=0), nn.ReLU())
        self.conv1d_approximator = nn.Conv1d(32, 1, 5, padding=0)
        self.dropout = nn.Dropout(0.6)
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
        logits = logits.permute(0, 2, 1)
        x = x.permute(0, 2, 1)  # b*t*c

        ## gcn
        scoadj = self.sadj(logits.detach(), seq_len)
        adj = self.adj(inputs, seq_len)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1_h = self.relu(self.gc1(x, adj))
        x1_h = self.dropout(x1_h)
        x2_h = self.relu(self.gc3(x, disadj))
        x2_h = self.dropout(x2_h)
        x3_h = self.relu(self.gc5(x, scoadj))
        x3_h = self.dropout(x3_h)
        x1 = self.relu(self.gc2(x1_h, adj))
        x1 = self.dropout(x1)
        x2 = self.relu(self.gc4(x2_h, disadj))
        x2 = self.dropout(x2)
        x3 = self.relu(self.gc6(x3_h, scoadj))
        x3 = self.dropout(x3)
        x = torch.cat((x1, x2, x3), 2)
        x = self.classifier(x)
        return x, logits

    def sadj(self, logits, seq_len):
        lens = logits.shape[1]
        soft = nn.Softmax(1)
        logits2 = self.sigmoid(logits).repeat(1, 1, lens)
        tmp = logits2.permute(0, 2, 1)
        adj = 1. - torch.abs(logits2 - tmp)
        self.sig = lambda x:1/(1+torch.exp(-((x-0.5))/0.1))
        adj = self.sig(adj)
        output = torch.zeros_like(adj)
        if seq_len is None:
            for i in range(logits.shape[0]):
                tmp = adj[i]
                adj2 = soft(tmp)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = adj[i, :seq_len[i], :seq_len[i]]
                adj2 = soft(tmp)
                output[i, :seq_len[i], :seq_len[i]] = adj2
        return output


    def adj(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0,2,1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0,2,1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output


