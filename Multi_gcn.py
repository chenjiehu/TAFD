import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn as nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
import numpy as np
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta

        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法

        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        support = torch.mm(input_feature, self.weight)
        output = torch.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output


class Graphprogate(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta

        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(Graphprogate, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法

        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        output = torch.mm(adjacency, input_feature)
        if self.use_bias:
            output += self.bias
        return output

#改进四：多图GCN A+A^2最终版本
class MultiGCN_512(nn.Module):
    def __init__(self, input_dim, N_way):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(512)

        self.gcn = GraphConvolution(input_dim, 512)

        self.aifa1 = nn.Parameter(torch.Tensor(1), requires_grad=False)
        self.aifa2 = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.aifa3 = nn.Parameter(torch.Tensor(1), requires_grad=True)

        self.weight = Parameter(torch.FloatTensor(input_dim, 512))
        self.aifa1.data.fill_(0)
        self.aifa2.data.fill_(0)
        self.aifa3.data.fill_(0)


        self.test_N_way = N_way
        self.reset_parameters_kaiming()
    def forward(self,features):

        A = self.MultiAdjacencyCompute(features)
        x = self.gcn(A, features)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, 0.6, training=self.training)
        return x

    def MultiAdjacencyCompute(self,features):
        N = features.size(0)
        temp = torch.norm(features.repeat(N, 1) - features.repeat(1, N).view(N * N, -1), dim=1)
        adjacency_e = torch.exp(-temp.pow(2) / 9).view(N, N)
        _, position = torch.topk(adjacency_e, round(N / (self.test_N_way)), dim=1, sorted=False, out=None, largest=True)
        adjacency0 = torch.zeros(N, N).cuda()
        D_adjacency_e = torch.zeros(N,N).cuda()
        for num in range(N):        #保留每行最大的K歌元素
            adjacency0[num, position[num,:]] = 1
            adjacency0[num,num] = 0
        adjacency_e = torch.mul(adjacency0,adjacency_e)

        adjacency = torch.eye(N).cuda() + adjacency_e

        d = torch.sum(adjacency,dim=1)
        d = d + 1
        d = torch.sqrt(d)
        D = torch.diag(d)
        inv_D = torch.inverse(D)
        adjacencyn = torch.mm(torch.mm(inv_D, adjacency),inv_D)

        data = 0.5

        aifa = F.softmax(torch.cat([self.aifa1,self.aifa2,self.aifa3],dim=0),dim=0)

        adjacency = aifa[0]*torch.eye(N).cuda() + aifa[1]*adjacencyn + aifa[2]*torch.mm(adjacencyn,adjacencyn)

        return adjacency

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')

#改进四：多图GCN A+A^2最终版本
class MultiGCN(nn.Module):
    def __init__(self, input_dim, N_way):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(32)

        self.gcn = GraphConvolution(input_dim, 32)

        self.aifa1 = nn.Parameter(torch.Tensor(1), requires_grad=False)
        self.aifa2 = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.aifa3 = nn.Parameter(torch.Tensor(1), requires_grad=True)

        self.weight = Parameter(torch.FloatTensor(input_dim, 32))
        self.aifa1.data.fill_(0)
        self.aifa2.data.fill_(0)
        self.aifa3.data.fill_(0)


        self.test_N_way = N_way
        self.reset_parameters_kaiming()
    def forward(self,features):

        A = self.MultiAdjacencyCompute(features)
        x = self.gcn(A, features)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, 0.6, training=self.training)
        return x

    def MultiAdjacencyCompute(self,features):
        N = features.size(0)
        temp = torch.norm(features.repeat(N, 1) - features.repeat(1, N).view(N * N, -1), dim=1)
        adjacency_e = torch.exp(-temp.pow(2) / 9).view(N, N)
        _, position = torch.topk(adjacency_e, 3, dim=1, sorted=False, out=None, largest=True)
        # _, position = torch.topk(adjacency_e, round(N / (self.test_N_way)), dim=1, sorted=False, out=None, largest=True)
        adjacency0 = torch.zeros(N, N).cuda()
        D_adjacency_e = torch.zeros(N,N).cuda()
        for num in range(N):        #保留每行最大的K歌元素
            adjacency0[num, position[num,:]] = 1
            adjacency0[num,num] = 0
        adjacency_e = torch.mul(adjacency0,adjacency_e)

        adjacency = torch.eye(N).cuda() + adjacency_e

        d = torch.sum(adjacency,dim=1)
        d = d + 1
        d = torch.sqrt(d)
        D = torch.diag(d)
        inv_D = torch.inverse(D)
        adjacencyn = torch.mm(torch.mm(inv_D, adjacency),inv_D)

        data = 0.5

        aifa = F.softmax(torch.cat([self.aifa1,self.aifa2,self.aifa3],dim=0),dim=0)

        adjacency = aifa[0]*torch.eye(N).cuda() + aifa[1]*adjacencyn + aifa[2]*torch.mm(adjacencyn,adjacencyn)

        return adjacency

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')


#改进四：多图GCN A+A^2最终版本,_undirected graph
class MGCN(nn.Module):
    def __init__(self, input_dim, N_way):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(1000)

        self.gcn = GraphConvolution(input_dim, 1000)

        self.aifa1 = nn.Parameter(torch.Tensor(1), requires_grad=False)
        self.aifa2 = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.aifa3 = nn.Parameter(torch.Tensor(1), requires_grad=True)

        self.weight = Parameter(torch.FloatTensor(input_dim, 1000))
        self.aifa1.data.fill_(0)
        self.aifa2.data.fill_(0)
        self.aifa3.data.fill_(0)


        self.test_N_way = N_way
        self.reset_parameters_kaiming()
    def forward(self,features):
        # A = self.MultiAdjacencyCompute(features)
        # #x = self.bn1(torch.mm(A,features))
        # x = self.bn1(torch.mm(A,features))
        #
        # x = F.relu(x)

        A = self.MultiAdjacencyCompute(features)
#        x = self.bn2(torch.mm(A,x))
        x = self.gcn(A, features)
        #x = torch.mm(A,x)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, 0.6, training=self.training)
        return x

    def MultiAdjacencyCompute(self,features):
        N = features.size(0)
        temp = torch.norm(features.repeat(N, 1) - features.repeat(1, N).view(N * N, -1), dim=1)
        adjacency_e = torch.exp(-temp.pow(2) / 9).view(N, N)
        _, position = torch.topk(adjacency_e, round(N / (self.test_N_way)), dim=1, sorted=False, out=None, largest=True)
        adjacency0 = torch.zeros(N, N).cuda()
        D_adjacency_e = torch.zeros(N,N).cuda()
        for num in range(N):        #保留每行最大的K歌元素
            adjacency0[num, position[num,:]] = 1
            adjacency0[num,num] = 0
        adjacency0_T = adjacency0.t()
        adjacency0 = torch.mul(adjacency0, adjacency0_T)

        adjacency0_sum1 = torch.sum(adjacency0,dim=0)
        adjacency0_sum2 = torch.sum(adjacency0,dim=1)

        adjacency_e = torch.mul(adjacency0,adjacency_e)

        adjacency = torch.eye(N).cuda() + adjacency_e

        d = torch.sum(adjacency,dim=1)
        d = d + 1
        d = torch.sqrt(d)
        D = torch.diag(d)
        inv_D = torch.inverse(D)
        adjacencyn = torch.mm(torch.mm(inv_D, adjacency),inv_D)

        data = 0.5

        aifa = F.softmax(torch.cat([self.aifa1,self.aifa2,self.aifa3],dim=0),dim=0)

        adjacency = aifa[0]*torch.eye(N).cuda() + aifa[1]*adjacencyn + aifa[2]*torch.mm(adjacencyn,adjacencyn)

        adjacency0_sum1 = torch.sum(adjacency,dim=0)
        adjacency0_sum2 = torch.sum(adjacency,dim=1)

        return adjacency

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')

#改进四：多图GCN A+A^2最终版本
class MultiGCN_temp(nn.Module):
    def __init__(self, input_dim, N_way):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(1000)

        self.gcn = GraphConvolution(input_dim, 1000)

        self.aifa1 = nn.Parameter(torch.Tensor(1), requires_grad=False)
        self.aifa2 = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.aifa3 = nn.Parameter(torch.Tensor(1), requires_grad=True)

        self.weight = Parameter(torch.FloatTensor(input_dim, 1000))
        self.aifa1.data.fill_(0)
        self.aifa2.data.fill_(0)
        self.aifa3.data.fill_(0)


        self.test_N_way = N_way
        self.reset_parameters_kaiming()
    def forward(self,features):
        # A = self.MultiAdjacencyCompute(features)
        # #x = self.bn1(torch.mm(A,features))
        # x = self.bn1(torch.mm(A,features))
        #
        # x = F.relu(x)

        A = self.MultiAdjacencyCompute(features)
#        x = self.bn2(torch.mm(A,x))
        x = self.gcn(A, features)
        #x = torch.mm(A,x)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, 0.6, training=self.training)
        return x

    def MultiAdjacencyCompute(self,features):
        N = features.size(0)
        temp = torch.norm(features.repeat(N, 1) - features.repeat(1, N).view(N * N, -1), dim=1)
        adjacency_e = torch.exp(-temp.pow(2) / 9).view(N, N)
        _, position = torch.topk(adjacency_e, round(N / (self.test_N_way)), dim=1, sorted=False, out=None, largest=True)
        adjacency0 = torch.zeros(N, N).cuda()
        D_adjacency_e = torch.zeros(N,N).cuda()
        for num in range(N):        #保留每行最大的K歌元素
            adjacency0[num, position[num,:]] = 1
            adjacency0[num,num] = 0
        adjacency_e = torch.mul(adjacency0,adjacency_e)

        adjacency = torch.eye(N).cuda() + adjacency_e

        d = torch.sum(adjacency,dim=1)
        d = d + 1
        d = torch.sqrt(d)
        D = torch.diag(d)
        inv_D = torch.inverse(D)
        adjacencyn = torch.mm(torch.mm(inv_D, adjacency),inv_D)

        data = 0.5

        aifa = F.softmax(torch.cat([self.aifa1,self.aifa2,self.aifa3],dim=0),dim=0)

        adjacency = aifa[1]*adjacencyn + aifa[2]*torch.mm(adjacencyn,adjacencyn)

        return adjacency

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')

#改进四：多图GCN A+A^2最终版本
class MultiGCN_temp2(nn.Module):
    def __init__(self, input_dim, N_way):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(1000)

        self.gcn = GraphConvolution(input_dim, 1000)

        self.aifa1 = nn.Parameter(torch.Tensor(1), requires_grad=False)
        self.aifa2 = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.aifa3 = nn.Parameter(torch.Tensor(1), requires_grad=True)

        self.weight = Parameter(torch.FloatTensor(input_dim, 1000))
        self.aifa1.data.fill_(0)
        self.aifa2.data.fill_(0)
        self.aifa3.data.fill_(0)


        self.test_N_way = N_way
        self.reset_parameters_kaiming()
    def forward(self,features):
        # A = self.MultiAdjacencyCompute(features)
        # #x = self.bn1(torch.mm(A,features))
        # x = self.bn1(torch.mm(A,features))
        #
        # x = F.relu(x)

        A = self.MultiAdjacencyCompute(features)
#        x = self.bn2(torch.mm(A,x))
        x = self.gcn(A, features)
        #x = torch.mm(A,x)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, 0.6, training=self.training)
        return x

    def MultiAdjacencyCompute(self,features):
        N = features.size(0)
        temp = torch.norm(features.repeat(N, 1) - features.repeat(1, N).view(N * N, -1), dim=1)
        adjacency_e = torch.exp(-temp.pow(2) / 9).view(N, N)
        _, position = torch.topk(adjacency_e, round(N / (self.test_N_way)), dim=1, sorted=False, out=None, largest=True)
        adjacency0 = torch.zeros(N, N).cuda()
        D_adjacency_e = torch.zeros(N,N).cuda()
        for num in range(N):        #保留每行最大的K歌元素
            adjacency0[num, position[num,:]] = 1
            adjacency0[num,num] = 0
        adjacency_e = torch.mul(adjacency0,adjacency_e)

        adjacency = torch.eye(N).cuda() + adjacency_e

        d = torch.sum(adjacency,dim=1)
        d = d + 1
        d = torch.sqrt(d)
        D = torch.diag(d)
        inv_D = torch.inverse(D)
        adjacencyn = torch.mm(torch.mm(inv_D, adjacency),inv_D)

        data = 0.5

        aifa = F.softmax(torch.cat([self.aifa1,self.aifa2,self.aifa3],dim=0),dim=0)

        adjacency = torch.eye(N).cuda()

        return adjacency

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')


#改进四：ProtoGCN最终版本
class ProtoGCN(nn.Module):
    def __init__(self, input_dim, N_way):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(1000)

        self.gcn = GraphConvolution(input_dim, 1000)

        self.weight = Parameter(torch.FloatTensor(input_dim, 1000))

        self.test_N_way = N_way
        self.reset_parameters_kaiming()
    def forward(self,features):
        A = self.MultiAdjacencyCompute(features)
        #x = self.bn1(torch.mm(A,features))
        # x = self.bn1(torch.mm(A,features))
        #
        # x = F.relu(x)
        #
        # A = self.MultiAdjacencyCompute(x)
#        x = self.bn2(torch.mm(A,x))
        x = self.gcn(A, features)
        #x = torch.mm(A,x)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, 0.6, training=self.training)
        return x

    def MultiAdjacencyCompute(self,features):
        N = features.size(0)
        temp = torch.norm(features.repeat(N, 1) - features.repeat(1, N).view(N * N, -1), dim=1)
        adjacency_e = torch.exp(-temp.pow(2) / 9).view(N, N)
        _, position = torch.topk(adjacency_e, round(N / self.test_N_way), dim=1, sorted=False, out=None)
        adjacency0 = torch.zeros(N, N).cuda()
        D_adjacency_e = torch.zeros(N,N).cuda()
        for num in range(N):        #保留每行最大的K歌元素
            adjacency0[num, position[num,:]] = 1
            adjacency0[num,num] = 0
        adjacency_e = torch.mul(adjacency0,adjacency_e)

        adjacency = torch.eye(N).cuda() + adjacency_e

        d = torch.sum(adjacency,dim=1)
        d = d + 1
        d = torch.sqrt(d)
        D = torch.diag(d)
        inv_D = torch.inverse(D)
        adjacency = torch.mm(torch.mm(inv_D, adjacency),inv_D)


        return adjacency

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')

#改进四：ProtoIGCN最终版本
class ProtoIGCN(nn.Module):
    def __init__(self, input_dim, N_way):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(1000)
        # self.bn3 = nn.BatchNorm1d(input_dim)
        # self.bn4 = nn.BatchNorm1d(1600)

        self.gcn = GraphConvolution(input_dim, 1000)
        # self.sigama1 = nn.Parameter(torch.Tensor(1), requires_grad=False)
        # self.sigama2 = nn.Parameter(torch.Tensor(1), requires_grad=False)


        self.weight = Parameter(torch.FloatTensor(input_dim, 1000))



        self.test_N_way = N_way
        self.reset_parameters_kaiming()
    def forward(self,features):
        A = self.MultiAdjacencyCompute(features)
        #x = self.bn1(torch.mm(A,features))
        x = self.bn1(torch.mm(A,features))

        x = F.relu(x)

        A = self.MultiAdjacencyCompute(x)
#        x = self.bn2(torch.mm(A,x))
        x = self.gcn(A, x)
        #x = torch.mm(A,x)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, 0.6, training=self.training)
        return x

    def MultiAdjacencyCompute(self,features):
        N = features.size(0)
        temp = torch.norm(features.repeat(N, 1) - features.repeat(1, N).view(N * N, -1), dim=1)
        adjacency_e = torch.exp(-temp.pow(2) / 9).view(N, N)
        _, position = torch.topk(adjacency_e, round(N / (self.test_N_way)), dim=1, sorted=False, out=None)
        adjacency0 = torch.zeros(N, N).cuda()
        D_adjacency_e = torch.zeros(N,N).cuda()
        for num in range(N):        #保留每行最大的K歌元素
            adjacency0[num, position[num,:]] = 1
            adjacency0[num,num] = 0
        adjacency_e = torch.mul(adjacency0,adjacency_e)

        adjacency = torch.eye(N).cuda() + adjacency_e

        d = torch.sum(adjacency,dim=1)
        d = d + 1
        d = torch.sqrt(d)
        D = torch.diag(d)
        inv_D = torch.inverse(D)
        adjacency = torch.mm(torch.mm(inv_D, adjacency),inv_D)
        adjacency = torch.mm(adjacency,adjacency)

        return adjacency

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')

#改进四：ProtoGCN最终版本
class GNN(nn.Module):
    def __init__(self, input_dim, N_way):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(1000)

        self.gnn = Graphprogate(input_dim, 1000)

        self.test_N_way = N_way
    def forward(self,features):
        A = self.MultiAdjacencyCompute(features)
        #x = self.bn1(torch.mm(A,features))
        # x = self.bn1(torch.mm(A,features))
        #
        # x = F.relu(x)
        #
        # A = self.MultiAdjacencyCompute(x)
#        x = self.bn2(torch.mm(A,x))
        x = self.gnn(A, features)
        #x = torch.mm(A,x)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, 0.6, training=self.training)
        return x

    def MultiAdjacencyCompute(self,features):
        N = features.size(0)
        temp = torch.norm(features.repeat(N, 1) - features.repeat(1, N).view(N * N, -1), dim=1)
        adjacency_e = torch.exp(-temp.pow(2) / 9).view(N, N)
        _, position = torch.topk(adjacency_e, round(N / self.test_N_way), dim=1, sorted=False, out=None)
        adjacency0 = torch.zeros(N, N).cuda()
        D_adjacency_e = torch.zeros(N,N).cuda()
        for num in range(N):        #保留每行最大的K歌元素
            adjacency0[num, position[num,:]] = 1
            adjacency0[num,num] = 0
        adjacency_e = torch.mul(adjacency0,adjacency_e)

        adjacency = torch.eye(N).cuda() + adjacency_e

        d = torch.sum(adjacency,dim=1)
        d = d + 1
        d = torch.sqrt(d)
        D = torch.diag(d)
        inv_D = torch.inverse(D)
        adjacency = torch.mm(torch.mm(inv_D, adjacency),inv_D)


        return adjacency

class RelationNetwork(nn.Module):
    """Graph Construction Module"""

    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1))

        self.fc3 = nn.Linear(2 * 2, 8)
        self.fc4 = nn.Linear(8, 1)

        self.m0 = nn.MaxPool2d(2)  # max-pool without padding
        self.m1 = nn.MaxPool2d(2, padding=1)  # max-pool with padding

    def forward(self, x):
        x = x.view(-1, 64, 5, 5)

        out = self.layer1(x)
        out = self.layer2(out)
        # flatten
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc3(out))
        out = self.fc4(out)  # no relu

        out = out.view(out.size(0), -1)  # bs*1

        return out

#改进四：多图GCN A+A^2最终版本, sigma is trained
eps = np.finfo(float).eps
class MultiGCN_sigma(nn.Module):
    def __init__(self, input_dim, N_way):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(1000)

        self.gcn = GraphConvolution(input_dim, 1000)

        self.aifa1 = nn.Parameter(torch.Tensor(1), requires_grad=False)
        self.aifa2 = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.aifa3 = nn.Parameter(torch.Tensor(1), requires_grad=True)

        self.weight = Parameter(torch.FloatTensor(input_dim, 1000))
        self.aifa1.data.fill_(0)
        self.aifa2.data.fill_(0)
        self.aifa3.data.fill_(0)


        self.test_N_way = N_way
        self.sigma = 3
        self.relation = RelationNetwork()
        self.reset_parameters_kaiming()
    def forward(self,features):
        # A = self.MultiAdjacencyCompute(features)
        # #x = self.bn1(torch.mm(A,features))
        # x = self.bn1(torch.mm(A,features))
        #
        # x = F.relu(x)

        A = self.MultiAdjacencyCompute(features)
#        x = self.bn2(torch.mm(A,x))
        x = self.gcn(A, features)
        #x = torch.mm(A,x)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, 0.6, training=self.training)
        return x

    def MultiAdjacencyCompute(self,features):
        N = features.size(0)
        self.sigma = self.relation(features)
        emb_all = features / (self.sigma + eps)  # N*d
        emb1 = torch.unsqueeze(emb_all, 1)  # N*1*d
        emb2 = torch.unsqueeze(emb_all, 0)  # 1*N*d
        W = ((emb1 - emb2) ** 2).mean(2)  # N*N*d -> N*N
        adjacency_e = torch.exp(-W / 2)
        #temp = torch.norm(features.repeat(N, 1) - features.repeat(1, N).view(N * N, -1), dim=1)
        #sigma_2 = (self.sigma + eps)*(self.sigma + eps)
        #adjacency_e = torch.exp(-temp.pow(2) / sigma_2).view(N, N)
        _, position = torch.topk(adjacency_e, round(N / (self.test_N_way)), dim=1, sorted=False, out=None)
        adjacency0 = torch.zeros(N, N).cuda()
        D_adjacency_e = torch.zeros(N,N).cuda()
        for num in range(N):        #保留每行最大的K歌元素
            adjacency0[num, position[num,:]] = 1
            adjacency0[num,num] = 0
        adjacency_e = torch.mul(adjacency0,adjacency_e)

        adjacency = torch.eye(N).cuda() + adjacency_e

        d = torch.sum(adjacency,dim=1)
        d = d + 1
        d = torch.sqrt(d)
        D = torch.diag(d)
        inv_D = torch.inverse(D)
        adjacencyn = torch.mm(torch.mm(inv_D, adjacency),inv_D)

        data = 0.5

        aifa = F.softmax(torch.cat([self.aifa1,self.aifa2,self.aifa3]))

        adjacency = aifa[0]*torch.eye(N).cuda() + aifa[1]*adjacencyn + aifa[2]*torch.mm(adjacencyn,adjacencyn)

        return adjacency

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')

#改进四：多图GCN A+A^2最终版本，add with label progation
class MultiGCN_progation(nn.Module):
    def __init__(self, input_dim, N_way):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(1000)

        self.gcn = GraphConvolution(input_dim, 1000)

        self.aifa1 = nn.Parameter(torch.Tensor(1), requires_grad=False)
        self.aifa2 = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.aifa3 = nn.Parameter(torch.Tensor(1), requires_grad=True)

        self.alpha = nn.Parameter(torch.tensor(0.5).cuda(0), requires_grad=True)

        self.weight = Parameter(torch.FloatTensor(input_dim, 1000))
        self.aifa1.data.fill_(0)
        self.aifa2.data.fill_(0)
        self.aifa3.data.fill_(0)


        self.test_N_way = N_way
        self.reset_parameters_kaiming()
    def forward(self,features, s_label):
        # A = self.MultiAdjacencyCompute(features)
        # #x = self.bn1(torch.mm(A,features))
        # x = self.bn1(torch.mm(A,features))
        #
        # x = F.relu(x)

        A = self.MultiAdjacencyCompute(features)
#        x = self.bn2(torch.mm(A,x))
        x = self.gcn(A, features)
        #x = torch.mm(A,x)
        # x = F.relu(self.bn2(x))
        # x = F.dropout(x, 0.6, training=self.training)
        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        A = self.AdjacencyCompute(x)
        pred_label = self.progation_label(x,s_label)

        return pred_label

    def MultiAdjacencyCompute(self,features):
        N = features.size(0)
        temp = torch.norm(features.repeat(N, 1) - features.repeat(1, N).view(N * N, -1), dim=1)
        adjacency_e = torch.exp(-temp.pow(2) / 9).view(N, N)
        _, position = torch.topk(adjacency_e, round(N / (self.test_N_way)), dim=1, sorted=False, out=None)
        adjacency0 = torch.zeros(N, N).cuda()
        D_adjacency_e = torch.zeros(N,N).cuda()
        for num in range(N):        #保留每行最大的K歌元素
            adjacency0[num, position[num,:]] = 1
            adjacency0[num,num] = 0
        adjacency_e = torch.mul(adjacency0,adjacency_e)

        adjacency = torch.eye(N).cuda() + adjacency_e

        d = torch.sum(adjacency,dim=1)
        d = torch.sqrt(d)
        D = torch.diag(d)
        inv_D = torch.inverse(D)
        adjacencyn = torch.mm(torch.mm(inv_D, adjacency),inv_D)

        data = 0.5

        aifa = F.softmax(torch.cat([self.aifa1,self.aifa2,self.aifa3]))

        adjacency = aifa[0]*torch.eye(N).cuda() + aifa[1]*adjacencyn + aifa[2]*torch.mm(adjacencyn,adjacencyn)

        return adjacency

    def AdjacencyCompute(self,features):
        N = features.size(0)
        temp = torch.norm(features.repeat(N, 1) - features.repeat(1, N).view(N * N, -1), dim=1)
        adjacency_e = torch.exp(-temp.pow(2) / 30).view(N, N)
        _, position = torch.topk(adjacency_e, round(N / (self.test_N_way)), dim=1, sorted=False, out=None)
        adjacency0 = torch.zeros(N, N).cuda()
        D_adjacency_e = torch.zeros(N,N).cuda()
        for num in range(N):        #保留每行最大的K歌元素
            adjacency0[num, position[num,:]] = 1
            adjacency0[num,num] = 0
        adjacency_e = torch.mul(adjacency0,adjacency_e)

        adjacency = torch.eye(N).cuda() + adjacency_e

        d = torch.sum(adjacency,dim=1)
        d = torch.sqrt(d)
        D = torch.diag(d)
        inv_D = torch.inverse(D)
        adjacencyn = torch.mm(torch.mm(inv_D, adjacency),inv_D)
        return adjacency

    def progation_label(self, features, s_labels):
        eps = np.finfo(float).eps

        num_classes = len(np.unique(s_labels.cpu()))
        num_support = int(s_labels.shape[0] / num_classes)
        len_query = features.shape[0] - s_labels.shape[0]

        s_labels = s_labels.unsqueeze(dim=1)
        temp_labels = torch.zeros(num_support*num_classes,num_classes).cuda()
        s_labels = temp_labels.scatter_(1,s_labels,1)

        S = self.AdjacencyCompute(features)

        ys = s_labels
        yu = torch.zeros(len_query, num_classes).cuda(0)
        # yu = (torch.ones(num_classes*num_queries, num_classes)/num_classes).cuda(0)
        y = torch.cat((ys, yu), 0)
        N = S.shape[0]
        F_all = torch.matmul(torch.inverse(torch.eye(N).cuda(0) - self.alpha * S + eps), y)
        Fq = F_all[num_classes * num_support:, :]  # query predictions

        return F_all, Fq

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def cos_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    ab = torch.mul(a,b)
    ab = torch.sum(ab, dim=2)
    a_norm = torch.norm(a,dim=2)
    b_norm = torch.norm(b,dim=2)
    ab_norm = torch.mul(a_norm,b_norm)
    logits = ab/ab_norm
    return logits


if __name__=='__main__':
    model = GNN(input_dim=10, N_way=10).cuda()
    model_MGNN = MultiGCN(input_dim=10, N_way=10).cuda()
    feature = torch.randn(200,1600).cuda()
    feature1 = model(feature)
    feature2 = model_MGNN(feature)

    print('# generator parameters:', sum(param.numel() for param in model.parameters()))

    print('# discriminator parameters:', sum(param.numel() for param in model_MGNN.parameters()))
