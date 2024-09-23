""
# Fewshot Semantic Segmentation the A li learned by learning
import models_RS

""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .vgg import Encoder
# from archs import Encoder
from skimage.segmentation import slic,mark_boundaries
from Multi_gcn import euclidean_metric, cos_metric
# from ResNetBackbone import resnet50 as Encoder
# from ResNetBackbone import Classifier_Module
from Multi_gcn import GraphConvolution
from torch.nn.parameter import Parameter
from relation_network import RelationNetwork
from model.Encoder import encoder_fuse, encoder_fuse8, encoder_fuse14, encoder_fuse12
import time

class MultiGCN_relation(nn.Module):
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

        self.relation = RelationNetwork()

        self.test_N_way = N_way
        self.reset_parameters_kaiming()
    def forward(self,features):

        A = self.AdjacencyCompute(features)

        x = self.gcn(A, features)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, 0.6, training=self.training)
        return x, A

    def AdjacencyCompute(self,features):
        N = features.size(0)
        temp1 = features.repeat(N, 1)
        temp2 = features.repeat(1, N).view(N * N, -1)
        temp = torch.cat([temp1,temp2], dim = 1)
        adjacency = self.relation(temp)
        adjacency = torch.reshape(adjacency,(N,N))
        adjacency = torch.eye(N).cuda() + adjacency

        d = torch.sum(adjacency, dim=1)
        d = d + 1
        d = torch.sqrt(d)
        D = torch.diag(d)
        inv_D = torch.inverse(D)
        adjacency = torch.mm(torch.mm(inv_D, adjacency), inv_D)

        # aifa = F.softmax(torch.cat([self.aifa1, self.aifa2, self.aifa3], dim=0), dim=0)
        #
        # adjacency = aifa[0] * torch.eye(N).cuda() + aifa[1] * adjacencyn + aifa[2] * torch.mm(adjacencyn, adjacencyn)

        return adjacency

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
class Feature_extractor(nn.Module):
    """
    INPUT: image, label, slic_results
    OUTPUT: superpixel_fts, superpixel_label, feature map
    """
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None, args=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        self.way = args.way
        self.shot = args.shot
        # Encoder
        self.encoder = encoder_fuse14(layers=50, classes=6)

    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_maks, img_segments):

        self.size = qry_imgs.shape[3]

        ###### Extract features ######
        imgs_concat = torch.cat([supp_imgs, qry_imgs], dim=0)
        mask_concat = torch.cat([supp_mask, qry_maks], dim=0)

        img_fts = self.encoder(imgs_concat)

        feature_superpixel, label_superpixel_shot, label_superpixel_query = self.extract_superpixel_all(img_fts,
                                                                                                        mask_concat,
                                                                                                        img_segments)

        return feature_superpixel, label_superpixel_shot, label_superpixel_query, img_fts

    def extract_superpixel_all(self, img_fts, support_mask ,img_segments):
        """
        Compute the superpixels

        Args:
            img_fts: embedding features for images
                expect shape: N x C x H' x W'
            img_segments: SLIC seg results
                expect shape: N x H x W
        """
        feature_superpixel = []
        label_seperpixel_shot = []
        label_seperpixel_query = []
        num_images = img_fts.shape[0]
        fts_size = img_fts.shape[3]
        # img_segments = np.resize(img_segments, new_shape=fts_size)  cjh gai

        for i in range(self.shot):
            Number_seg = np.unique(img_segments[i])
            for j in range(len(Number_seg)):
                t1 = time.time()
                feature_temp = img_fts.permute(0,2,3,1)[i][np.where(img_segments[i] == Number_seg[j])]
                feature_temp = torch.mean(feature_temp,dim=0)
                label_temp = support_mask[i][np.where(img_segments[i] == Number_seg[j])]
                # label_temp = max(set(label_temp))
                label_temp = label_temp.tolist()
                if not len(label_temp):
                    continue
                label_temp = max(label_temp, key=label_temp.count)
                feature_superpixel.append(feature_temp)
                label_temp = torch.tensor(label_temp)
                label_seperpixel_shot.append(label_temp.cuda())
        for i in range(self.shot, num_images):
            Number_seg = np.unique(img_segments[i])
            for j in range(len(Number_seg)):
                feature_temp = img_fts.permute(0,2,3,1)[i][np.where(img_segments[i] == Number_seg[j])]
                feature_temp = torch.mean(feature_temp,dim=0)
                label_temp = support_mask[i][np.where(img_segments[i] == Number_seg[j])]
                # label_temp = max(set(label_temp))
                label_temp = label_temp.tolist()
                if not len(label_temp):
                    continue
                label_temp = max(label_temp, key=label_temp.count)
                feature_superpixel.append(feature_temp)
                label_temp = torch.tensor(label_temp)
                label_seperpixel_query.append(label_temp.cuda())

        feature_superpixel = torch.stack(feature_superpixel)
        label_seperpixel_shot = torch.stack(label_seperpixel_shot)
        label_seperpixel_query = torch.stack(label_seperpixel_query)
        return feature_superpixel, label_seperpixel_shot, label_seperpixel_query


    def getPrototype(self, fg_fts, fg_label):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: number_of_shot_superpixel x dim
            fg_label: lists of list of fg_labelfor each way/shot
                expect shape: number_of_shot_superpixel
        """
        fg_prototypes = torch.zeros(self.way, fg_fts.shape[1])
        lists = [[] for i in range(self.way)]
        length = min(fg_fts.shape[0], fg_label.shape[0])
        for i in range(length):
            lists[fg_label[i]].append(fg_fts[i])
        for i in range(self.way):
            if lists[i] != []:
                fg_prototypes[i] = torch.mean(torch.stack(lists[i]), dim=0)

        return fg_prototypes

    def label_to_mask(self, logits, segments):
        length = logits.shape[0]
        num_segments = segments.shape[0]
        num_superpixel = np.zeros(num_segments)
        out_put = torch.zeros(num_segments,self.size, self.size, self.way).cuda()
        index = 0
        for i in range(num_segments):
            num_superpixel[i] = len(np.unique(segments[i]))
        for i in range(num_segments):
            temp = np.unique(segments[i])
            for j in range(num_superpixel[i].astype(np.int32)):
                if index < logits.shape[0]:
                    out_put[i][np.where(segments[i] == temp[j])] = logits[index]
                    index = index+1
        return out_put

class MGCN_TPN_Classifier_union_gai(nn.Module):
    """
    INPUT: superpixel_fts, superpixel_label_shot
    OUTPUT: pred_results
    """
    def __init__(self, in_channels=3, cfg=None, args=None):
        super().__init__()
        self.config = cfg or {'align': False}
        self.way = args.way   #cjh
        self.shot = args.shot
        self.MGCN_TPN = models_RS.MGCN_LabelPropagation_A_TPN_Proto_gai(args=args)    #cjh yicuo

    def forward(self, feature_superpixel, label_superpixel_shot, label_superpixel_query, slic_npy, class_num):

        # self.way = np.unique(label_superpixel_shot.cpu().numpy())    #cjh
        self.size = slic_npy.shape[1]
        img_segments = slic_npy
# TPN TO PREDICT
        _, predict_p, pred_label, _, _, index_no_predicetd, index_predicetd = self.MGCN_TPN(feature_superpixel, label_superpixel_shot,
                                                       label_superpixel_query, class_num)


        used_label = torch.cat([label_superpixel_shot, pred_label], dim=0)
        shot_num = label_superpixel_shot.shape[0]
        shot_list = list(range(shot_num))
        # print("index_predicetd",len(index_predicetd))
        if index_predicetd.shape == torch.Size([]):
            index_union_predicted = shot_list
        else:
            index_union_predicted = [jj+shot_num for jj in index_predicetd]
            index_union_predicted = index_union_predicted + shot_list
        if index_no_predicetd.shape != torch.Size([]):
            index_union_unpredicted = [jj+shot_num for jj in index_no_predicetd]
    # PROTO TO PREDICT REMAIN
            proto = self.getPrototype(feature_superpixel[index_union_predicted], used_label[index_union_predicted], class_num)
            proto = proto.cuda()
            query_feature_unpred = feature_superpixel[index_union_unpredicted,:]
            logits = euclidean_metric(query_feature_unpred, proto)

            predict_p_out = torch.zeros_like(predict_p)
            predict_p_out[index_predicetd] = predict_p[index_predicetd]
            # predict_p_out[index_no_predicetd] = logits
            predict_p_out[index_no_predicetd] = logits
        else:
            predict_p_out = predict_p
        output = self.label_to_mask(predict_p_out, img_segments[self.shot:], class_num)

        return output, predict_p_out

    def label_to_mask(self, logits, segments, class_num):
        length = logits.shape[0]
        num_segments = segments.shape[0]
        num_superpixel = np.zeros(num_segments)
        out_put = torch.zeros(num_segments,self.size, self.size, class_num).cuda()
        index = 0
        for i in range(num_segments):
            num_superpixel[i] = len(np.unique(segments[i]))
        for i in range(num_segments):
            temp = np.unique(segments[i])
            for j in range(num_superpixel[i].astype(np.int32)):
                if index < logits.shape[0]:
                    out_put[i][np.where(segments[i] == temp[j])] = logits[index]
                    index = index+1
        return out_put

    def getPrototype(self, fg_fts, fg_label, class_num):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: number_of_shot_superpixel x dim
            fg_label: lists of list of fg_labelfor each way/shot
                expect shape: number_of_shot_superpixel
        """
        fg_prototypes = torch.zeros(class_num, fg_fts.shape[1])
        lists = [[] for i in range(class_num)]
        length = min(fg_fts.shape[0], fg_label.shape[0])
        for i in range(length):
            lists[fg_label[i]].append(fg_fts[i])
        for i in range(class_num):
            if lists[i] != []:
                fg_prototypes[i] = torch.mean(torch.stack(lists[i]), dim=0)

        return fg_prototypes

class MGCN_TPN_Classifier(nn.Module):
    """
    INPUT: superpixel_fts, superpixel_label_shot
    OUTPUT: pred_results
    """
    def __init__(self, in_channels=3, cfg=None, args=None):
        super().__init__()
        self.config = cfg or {'align': False}
        self.way = args.way
        self.shot = args.shot
        self.MGCN_TPN = models_RS.MGCN_LabelPropagation_A_TPN_Proto(args=args)    #cjh yicuo

    def forward(self, feature_superpixel, label_superpixel_shot, label_superpixel_query, slic_npy):

        self.size = slic_npy.shape[1]
        img_segments = slic_npy

        _, predict_p, pred_label, _, _,_,_ = self.MGCN_TPN(feature_superpixel, label_superpixel_shot,
                                                       label_superpixel_query)
        output = self.label_to_mask(predict_p, img_segments[self.shot:])

        return output, predict_p

    def label_to_mask(self, logits, segments):
        length = logits.shape[0]
        num_segments = segments.shape[0]
        num_superpixel = np.zeros(num_segments)
        out_put = torch.zeros(num_segments,self.size, self.size, self.way).cuda()
        index = 0
        for i in range(num_segments):
            num_superpixel[i] = len(np.unique(segments[i]))
        for i in range(num_segments):
            temp = np.unique(segments[i])
            for j in range(num_superpixel[i].astype(np.int32)):
                if index < logits.shape[0]:
                    out_put[i][np.where(segments[i] == temp[j])] = logits[index]
                    index = index+1
        return out_put

