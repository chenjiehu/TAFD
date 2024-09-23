# -------------------------------------
# Project: Transductive Propagation Network for Few-shot Learning
# Date: 2019.1.11
# Author: Yanbin Liu
# All Rights Reserved
# -------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score
# from Multi_gcn import MultiGCN, MultiGCN_A, MultiGCN_A2, MultiGCN1


class Averager_vector():

    def __init__(self,num):
        self.n = 0
        self.v = torch.zeros(num)

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
def cal_acc2(gt_list, predict_list, num):
    acc_sum = 0
    class_pred = torch.zeros(num)
    for n in range(num):
        y = []
        pred_y = []
        for i in range(gt_list.shape[0]):
            gt = gt_list[i]
            predict = predict_list[i]
            if gt == n:
                y.append(gt.cpu())
                pred_y.append(predict.cpu())
        acc = accuracy_score(y, pred_y)
        # print ('{}: {:4f}'.format(n if n != (num - 1) else 'Unk', acc))
        # if n == (num - 1):
        #     print ('Known Avg Acc: {:4f}'.format(acc_sum / (num - 1)))
        class_pred[n] = acc
    # print ('Avg Acc: {:4f}'.format(acc_sum / num))
    return class_pred

def compute_deviation(acc_list, commom_num):
    number = len(acc_list)
    acc_common = []
    acc_private = []
    acc_all = []
    for i in range(number):
        acc_temp = acc_list[i][:commom_num]
        acc_common.append(torch.mean(acc_temp))
        acc_temp = acc_list[i][commom_num:]
        acc_private.append(torch.mean(acc_temp))
        acc_temp = acc_list[i]
        acc_all.append(torch.mean(acc_temp))
    common_std = np.std(acc_common)
    private_std = np.std(acc_private)
    all_std = np.std(acc_all)
    common_mean = np.mean(acc_common)
    private_mean = np.mean(acc_private)
    all_mean = np.mean(acc_all)
    print('common std:',common_std)
    print('private std:', private_std)
    print('all_std:',all_std)

    print('common_acc:',common_mean)
    print('private_acc:',private_mean)
    print('all_mean:',all_mean)


class CNNEncoder(nn.Module):
    """Encoder for feature embedding"""

    def __init__(self, args):
        super(CNNEncoder, self).__init__()
        self.args = args
        h_dim, z_dim = args.h_dim, args.z_dim
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))

    def forward(self, x):
        """x: bs*3*84*84 """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


class RelationNetwork(nn.Module):
    """Graph Construction Module"""

    def __init__(self, input_dim):
        super(RelationNetwork, self).__init__()

        self.fc3 = nn.Linear(input_dim, 8)
        self.fc4 = nn.Linear(8, 1)

        self.m0 = nn.MaxPool2d(2)  # max-pool without padding
        self.m1 = nn.MaxPool2d(2, padding=1)  # max-pool with padding

    def forward(self, x, rn):
        # x = x.view(-1, 64, 5, 5)

        out = F.relu(self.fc3(x))
        out = self.fc4(out)  # no relu

        out = out.view(out.size(0), -1)  # bs*1

        return out


class RelationNetwork_cat(nn.Module):
    """Graph Construction Module"""

    def __init__(self):
        super(RelationNetwork_cat, self).__init__()

        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 32)

        self.m0 = nn.MaxPool2d(2)  # max-pool without padding
        self.m1 = nn.MaxPool2d(2, padding=1)  # max-pool with padding

    def forward(self, x, rn):
        N, d = x.shape[0], x.shape[1]
        em1 = x.repeat(N,1)
        em2 = x.repeat(1,N).view(N*N,-1)
        # x = x.view(-1, 64, 5, 5)
        em = torch.cat([em1,em2], dim=0)

        out = F.relu(self.fc3(em))
        out = self.fc4(out)  # no relu

        out = out.view(N, -1)  # bs*1
        out = torch.softmax(out,dim=0)

        return out


class LabelPropagation(nn.Module):
    """Label Propagation"""

    def __init__(self, args):
        super(LabelPropagation, self).__init__()
        # self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        # self.h_dim, self.z_dim = args['h_dim'], args['z_dim']

        self.args = args
        self.encoder = CNNEncoder(args)
        self.relation = RelationNetwork()

        if args.rn == 300:  # learned sigma, fixed alpha
            self.alpha = torch.tensor([args.alpha], requires_grad=False).cuda(0)
        elif args.rn == 30:  # learned sigma, learned alpha
            self.alpha = nn.Parameter(torch.tensor([args.alpha]).cuda(0), requires_grad=True)

    def forward(self, inputs, common_class_num):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        # init
        eps = np.finfo(float).eps

        [support, s_labels, query, q_labels] = inputs
        s_labels = s_labels.cpu()
        num_classes = len(np.unique(s_labels.cpu()))
        num_support = int(s_labels.shape[0] / num_classes)
        num_queries = int(query.shape[0] / num_classes)

        s_labels = s_labels.unsqueeze(dim=1)
        temp_labels = torch.zeros(num_support*num_classes,num_classes).cuda()
        s_labels = temp_labels.scatter_(1,s_labels.cuda(),1)

        q_labels = q_labels.unsqueeze(dim=1)
        #temp_labels = torch.zeros(num_queries*num_classes, num_classes).cuda()
        len_query = q_labels.size(0)
        temp_labels = torch.zeros(len_query, num_classes).cuda()
        q_labels = temp_labels.scatter_(1,q_labels,1)

        # Step1: Embedding
        inp = torch.cat((support, query), 0)
        emb_all = self.encoder(inp).view(-1, 1600)
        N, d = emb_all.shape[0], emb_all.shape[1]

        # Step2: Graph Construction
        ## sigmma
        if self.args.rn in [30, 300]:
            self.sigma = self.relation(emb_all, self.args.rn)

            ## W
            emb_all = emb_all / (self.sigma + eps)  # N*d
            emb1 = torch.unsqueeze(emb_all, 1)  # N*1*d
            emb2 = torch.unsqueeze(emb_all, 0)  # 1*N*d
            W = ((emb1 - emb2) ** 2).mean(2)  # N*N*d -> N*N
            W = torch.exp(-W / 2)

        ## keep top-k values
        if self.args.k > 0:
            topk, indices = torch.topk(W, self.args.k)
            mask = torch.zeros_like(W)
            mask = mask.scatter(1, indices, 1)
            mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  # union, kNN graph
            # mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
            W = W * mask

        ## normalize
        D = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)  # 此处与muttiGCN计算不一样
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)
        S = D1 * W * D2

        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        ys = s_labels
        yu = torch.zeros(len_query, num_classes).cuda(0)
        # yu = (torch.ones(num_classes*num_queries, num_classes)/num_classes).cuda(0)
        y = torch.cat((ys, yu), 0)
        F_all = torch.matmul(torch.inverse(torch.eye(N).cuda(0) - self.alpha * S + eps), y)
        Fq = F_all[num_classes * num_support:, :]  # query predictions

        # Step4: Cross-Entropy Loss
        ce = nn.CrossEntropyLoss().cuda(0)
        ## both support and query loss
        gt = torch.argmax(torch.cat((s_labels, q_labels), 0), 1)
        loss = ce(F_all, gt)
        ## acc
        predq = torch.argmax(Fq, 1)
        gtq = torch.argmax(q_labels, 1)
        correct = (predq == gtq).sum()
        total = len_query
        acc = 1.0 * correct.float() / float(total)
        query_feature = emb_all[num_classes * num_support:]
        predict_label = Fq.argmax(dim = 1)
        predict_p = F.softmax(Fq, dim=1)

        acc_every_class = cal_acc2(gtq, predict_label, num_classes)
        # compute_deviation([acc_every_class], common_class_num)
        # print('every class:',acc_every_class)

        return query_feature, predict_p, predict_label, loss, acc, acc_every_class



class MGCN_LabelPropagation_A(nn.Module):
    """Label Propagation"""

    def __init__(self, args):
        super(MGCN_LabelPropagation_A, self).__init__()
        # self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        # self.h_dim, self.z_dim = args['h_dim'], args['z_dim']

        self.args = args
        self.way = args.way
        # self.encoder = CNNEncoder(args)
        self.relation = RelationNetwork()
        # self.GCN_filter = MultiGCN_A(input_dim=1600, N_way=args.way).cuda()


        if args.rn == 300:  # learned sigma, fixed alpha
            self.alpha = torch.tensor([args.alpha], requires_grad=False).cuda(0)
        elif args.rn == 30:  # learned sigma, learned alpha
            self.alpha = nn.Parameter(torch.tensor([args.alpha]).cuda(0), requires_grad=True)

    def forward(self, inputs, shot_label, query_labels):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        # init
        eps = np.finfo(float).eps

        s_labels = shot_label

        shot_num = shot_label.shape[0]
        support  = inputs[:shot_num]
        query = inputs[shot_num:]

        num_classes = self.way

        s_labels = s_labels.unsqueeze(dim=1)
        # temp_labels = torch.zeros(shot_num, num_classes).cuda()   cjh xiugai
        temp_labels = torch.zeros(shot_num, self.way).cuda()   #cjh xiugai
        # print('s_label:', torch.min(s_labels),torch.max(s_labels))
        # print('temp_shape', temp_labels.shape)
        s_labels = temp_labels.scatter_(1,s_labels,1)

        q_labels = query_labels
        q_labels = q_labels.unsqueeze(dim=1)
        #temp_labels = torch.zeros(num_queries*num_classes, num_classes).cuda()
        len_query = q_labels.size(0)
        temp_labels = torch.zeros(len_query, self.way).cuda()
        # print('q_label:', torch.min(q_labels),torch.max(q_labels))
        # print('temp_shape', temp_labels.shape)
        q_labels = temp_labels.scatter_(1,q_labels,1)

        # Step1: Embedding
        inp = torch.cat((support, query), 0)
        # emb_all = self.encoder(inp).view(-1, 1600)  #cjh
        emb_all = inputs
        N, d = emb_all.shape[0], emb_all.shape[1]
        if self.args.rn in [30, 300]:
            # self.sigma = self.relation(emb_all, self.args.rn)  #cjh_temp
            self.sigma = 9

            ## W
            emb_all = emb_all / (self.sigma + eps)  # N*d
            emb1 = torch.unsqueeze(emb_all, 1)  # N*1*d
            emb2 = torch.unsqueeze(emb_all, 0)  # 1*N*d
            W = ((emb1 - emb2) ** 2).mean(2)  # N*N*d -> N*N
            W = torch.exp(-W / 2)

        ## keep top-k values
        if self.args.k > 0:
            topk, indices = torch.topk(W, self.args.k)
            mask = torch.zeros_like(W)
            mask = mask.scatter(1, indices, 1)
            mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  # union, kNN graph
            # mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
            W = W * mask

        ## normalize
        D = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)  # 此处与muttiGCN计算不一样
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)
        S = D1 * W * D2
        # emb_all = self.GCN_filter(emb_all, S)
        # Step2: Graph Construction
        ## sigmma


        # S_M = self.aifa1*torch.eye(N).cuda() + self.aifa2*S + self.aifa3*torch.mm(S,S)

        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        ys = s_labels
        yu = torch.zeros(len_query, num_classes).cuda(0)
        # yu = (torch.ones(num_classes*num_queries, num_classes)/num_classes).cuda(0)
        y = torch.cat((ys, yu), 0)
        F_all = torch.matmul(torch.inverse(torch.eye(N).cuda(0) - self.alpha * S + eps), y)
        Fq = F_all[shot_num:, :]  # query predictions

        # Step4: Cross-Entropy Loss
        ce = nn.CrossEntropyLoss().cuda(0)
        ## both support and query loss
        gt = torch.argmax(torch.cat((s_labels, q_labels), 0), 1)
        loss = ce(F_all, gt)
        ## acc
        predq = torch.argmax(Fq, 1)
        gtq = torch.argmax(q_labels, 1)
        correct = (predq == gtq).sum()
        total = len_query
        acc = 1.0 * correct.float() / float(total)
        query_feature = emb_all[shot_num:]
        predict_label = Fq.argmax(dim = 1)
        predict_p = F.softmax(Fq, dim=1)

        return query_feature, predict_p, predict_label, loss, acc


Ignore_label = 255

# tpn A IS NOT AFFECTED BY LABELS IMFORMANTION

class MGCN_LabelPropagation_A_TPN_Proto(nn.Module):
    """Label Propagation"""

    def __init__(self, args):
        super(MGCN_LabelPropagation_A_TPN_Proto, self).__init__()

        self.args = args
        self.way = args.way   #cjh
        self.relation = RelationNetwork(args.input_dim)

        if args.rn == 300:  # learned sigma, fixed alpha
            self.alpha = torch.tensor([args.alpha], requires_grad=False).cuda(0)
        elif args.rn == 30:  # learned sigma, learned alpha
            self.alpha = nn.Parameter(torch.tensor([args.alpha]).cuda(0), requires_grad=True)

    def forward(self, inputs, shot_label, query_labels, class_num):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        # init
        # self.way = len(np.unique(shot_label.cpu().numpy()))   #cjh
        eps = np.finfo(float).eps

        s_labels = shot_label

        shot_num = shot_label.shape[0]
        support  = inputs[:shot_num]
        query = inputs[shot_num:]

        num_classes = class_num

        s_labels = s_labels.unsqueeze(dim=1)
        # temp_labels = torch.zeros(shot_num, num_classes).cuda()   cjh xiugai
        temp_labels = torch.zeros(shot_num, class_num).cuda() #cjh xiugai
        position_ignore = torch.nonzero(s_labels == Ignore_label)
        s_labels[position_ignore] = 0

        s_labels = temp_labels.scatter_(1,s_labels,1)
        s_labels[position_ignore,0] = 0

        q_labels = query_labels
        q_labels = q_labels.unsqueeze(dim=1)
        #temp_labels = torch.zeros(num_queries*num_classes, num_classes).cuda()
        len_query = q_labels.size(0)
        temp_labels = torch.zeros(len_query, class_num).cuda()
        position_ignore = torch.nonzero(q_labels == Ignore_label)
        q_labels[position_ignore] = 0
        # print('max',torch.max(q_labels))
        # print('min', torch.min(q_labels))
        q_labels = temp_labels.scatter_(1,q_labels,1)
        q_labels[position_ignore, 0] = 0

        # Step1: Embedding
        inp = torch.cat((support, query), 0)
        # emb_all = self.encoder(inp).view(-1, 1600)  #cjh
        emb_all = inputs
        N, d = emb_all.shape[0], emb_all.shape[1]
        if self.args.rn in [30, 300]:
            self.sigma = self.relation(emb_all, self.args.rn)  #cjh_temp

            ## W
            emb_all = emb_all / (self.sigma + eps)  # N*d
            emb1 = torch.unsqueeze(emb_all, 1)  # N*1*d
            emb2 = torch.unsqueeze(emb_all, 0)  # 1*N*d
            W = ((emb1 - emb2) ** 2).mean(2)  # N*N*d -> N*N
            W = torch.exp(-W)
        zeros_m = torch.zeros_like(W)
        W_select = torch.where(W>0.9, W, zeros_m)

        # select index of  signal node
        index_no_predicetd = []
        index_predicetd = []
        for i in range(shot_num, N):
            if torch.sum(W_select[i,:]) == W_select[i,i]:   # cjh cjh  original
            # if torch.sum(W_select[i, shot_num:N]) == 0:  # cjh cjh
                index_no_predicetd.append(i-shot_num)
            else:
                index_predicetd.append(i-shot_num)

        ## normalize
        D = W_select.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)  # 此处与muttiGCN计算不一样
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)
        S = D1 * W_select * D2
        # emb_all = self.GCN_filter(emb_all, S)
        # Step2: Graph Construction
        ## sigmma


        # S_M = self.aifa1*torch.eye(N).cuda() + self.aifa2*S + self.aifa3*torch.mm(S,S)

        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        ys = s_labels
        yu = torch.zeros(len_query, num_classes).cuda(0)
        # yu = (torch.ones(num_classes*num_queries, num_classes)/num_classes).cuda(0)
        y = torch.cat((ys, yu), 0)
        F_all = torch.matmul(torch.inverse(torch.eye(N).cuda(0) - self.alpha * S + eps), y)
        Fq = F_all[shot_num:, :]  # query predictions
        Fq_norm = torch.norm(F_all, dim=1)
        # index_no_predicetd = torch.nonzero(Fq_norm < 1).squeeze()
        # index_predicetd = torch.nonzero(Fq_norm >= 1).squeeze()


        # Step4: Cross-Entropy Loss
        ce = nn.CrossEntropyLoss().cuda(0)
        ## both support and query loss
        gt = torch.argmax(torch.cat((s_labels, q_labels), 0), 1)
        loss = ce(F_all, gt)
        ## acc
        predq = torch.argmax(Fq, 1)
        gtq = torch.argmax(q_labels, 1)
        correct = (predq == gtq).sum()
        total = len_query
        acc = 1.0 * correct.float() / float(total)
        query_feature = emb_all[shot_num:]
        predict_label = Fq.argmax(dim = 1)
        predict_p = F.softmax(Fq, dim=1)

        return query_feature, predict_p, predict_label, loss, acc, index_no_predicetd, index_predicetd



# tpn A with labels. for nodes with different labels, set acjacent as 0, same labels set 1, the A computation is different

class MGCN_LabelPropagation_A_TPN_Proto_gai(nn.Module):
    """Label Propagation"""

    def __init__(self, args):
        super(MGCN_LabelPropagation_A_TPN_Proto_gai, self).__init__()

        self.args = args
        self.way = args.way   #cjh
        self.relation = RelationNetwork(input_dim = args.feature_dim)

        if args.rn == 300:  # learned sigma, fixed alpha
            self.alpha = torch.tensor([args.alpha], requires_grad=False).cuda(0)
        elif args.rn == 30:  # learned sigma, learned alpha
            self.alpha = nn.Parameter(torch.tensor([args.alpha]).cuda(0), requires_grad=True)

    def forward(self, inputs, shot_label, query_labels, class_num):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        # init
        # self.way = len(np.unique(shot_label.cpu().numpy()))   #cjh
        eps = np.finfo(float).eps
        thres = 0.8

        s_labels = shot_label

        shot_num = shot_label.shape[0]
        support  = inputs[:shot_num]
        query = inputs[shot_num:]

        num_classes = class_num

        s_labels = s_labels.unsqueeze(dim=1)
        # temp_labels = torch.zeros(shot_num, num_classes).cuda()   cjh xiugai
        temp_labels = torch.zeros(shot_num, class_num).cuda() #cjh xiugai
        position_ignore = torch.nonzero(s_labels == Ignore_label)
        s_labels[position_ignore] = 0

        # data_min_view = torch.min(s_labels)
        # print(data_min_view)
        # data_max_view = torch.max(s_labels)
        # print(data_max_view)


        s_labels = temp_labels.scatter_(1,s_labels,1)
        s_labels[position_ignore,0] = 0

        q_labels = query_labels
        q_labels = q_labels.unsqueeze(dim=1)
        #temp_labels = torch.zeros(num_queries*num_classes, num_classes).cuda()
        len_query = q_labels.size(0)
        temp_labels = torch.zeros(len_query, class_num).cuda()
        position_ignore = torch.nonzero(q_labels == Ignore_label)
        q_labels[position_ignore] = 0
        # print('max',torch.max(q_labels))
        # print('min', torch.min(q_labels))
        q_labels = temp_labels.scatter_(1,q_labels,1)
        q_labels[position_ignore, 0] = 0

        # Step1: Embedding
        inp = torch.cat((support, query), 0)
        # emb_all = self.encoder(inp).view(-1, 1600)  #cjh
        emb_all = inputs
        N, d = emb_all.shape[0], emb_all.shape[1]
        if self.args.rn in [30, 300]:
            self.sigma = self.relation(emb_all, self.args.rn)  #cjh_temp

            ## W
            emb_all = emb_all / (self.sigma + eps)  # N*d
            emb1 = torch.unsqueeze(emb_all, 1)  # N*1*d
            emb2 = torch.unsqueeze(emb_all, 0)  # 1*N*d
            W = ((emb1 - emb2) ** 2).mean(2)  # N*N*d -> N*N
            W = torch.exp(-W)
        zeros_m = torch.zeros_like(W)
        W_select = torch.where(W>0.8, W, zeros_m)

        # select index of  signal node
        index_no_predicetd = []
        index_predicetd = []
        # for i in range(shot_num, N):
        #     if torch.sum(W_select[i,:]) == W_select[i,i]:   # cjh cjh  original
        #     # if torch.sum(W_select[i, shot_num:N]) == 0:  # cjh cjh
        #         index_no_predicetd.append(i-shot_num)
        #     else:
        #         index_predicetd.append(i-shot_num)

        ## normalize
        D = W_select.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)  # 此处与muttiGCN计算不一样
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)
        S = D1 * W_select * D2
        # print("S:",S)
        # emb_all = self.GCN_filter(emb_all, S)
        # Step2: Graph Construction
        ## sigmma


        # S_M = self.aifa1*torch.eye(N).cuda() + self.aifa2*S + self.aifa3*torch.mm(S,S)

        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        ys = s_labels
        yu = torch.zeros(len_query, num_classes).cuda(0)
        # print('original len_query',len_query)
        # yu = (torch.ones(num_classes*num_queries, num_classes)/num_classes).cuda(0)
        y = torch.cat((ys, yu), 0)
        F_all = torch.matmul(torch.inverse(torch.eye(N).cuda(0) - self.alpha * S + eps), y)
        Fq = F_all[shot_num:, :]  # query predictions
        Fq_norm = torch.norm(Fq, dim=1)
        index_no_predicetd = torch.nonzero(Fq_norm < thres).squeeze()
        index_predicetd = torch.nonzero(Fq_norm >= thres).squeeze()


        # Step4: Cross-Entropy Loss
        ce = nn.CrossEntropyLoss().cuda(0)
        ## both support and query loss
        gt = torch.argmax(torch.cat((s_labels, q_labels), 0), 1)
        loss = ce(F_all, gt)
        ## acc
        predq = torch.argmax(Fq, 1)
        gtq = torch.argmax(q_labels, 1)
        correct = (predq == gtq).sum()
        total = len_query
        acc = 1.0 * correct.float() / float(total)
        query_feature = emb_all[shot_num:]
        predict_label = Fq.argmax(dim = 1)
        predict_p = F.softmax(Fq, dim=1)

        return query_feature, predict_p, predict_label, loss, acc, index_no_predicetd, index_predicetd