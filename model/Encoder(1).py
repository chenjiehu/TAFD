import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import time
import cv2

import model.resnet as models
import model.vgg as vgg_models
from backbone import ResNet18, ResNet10


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat
  
def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

class encoder_fuse(nn.Module):
    def __init__(self, layers = 50, classes = 6, \
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
                 pretrained=True, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8],
                 ):
        super(encoder_fuse, self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm
        self.criterion = criterion
        self.shot = shot
        self.ppm_scales = ppm_scales

        models.BatchNorm = BatchNorm

        print('INFO: Using ResNet {}'.format(layers))
        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                    resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256

        fea_dim = 1024 + 512

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.pyramid_bins = ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )

        factor = 1
        mask_add_num = 1
        self.init_merge = []
        self.beta_conv = []
        self.inner_cls = []
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))
        self.init_merge = nn.ModuleList(self.init_merge)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)

        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim * len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins) - 1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))
        self.alpha_conv = nn.ModuleList(self.alpha_conv)

    def forward(self, x):
        x_size = x.size()
        # assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = x_size[2]
        w = x_size[3]
        # h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        # w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        #   Query Feature
        with torch.no_grad():
            feat_0 = self.layer0(x)
            feat_1 = self.layer1(feat_0)
            feat_2 = self.layer2(feat_1)
            feat_3 = self.layer3(feat_2)
            feat_4 = self.layer4(feat_3)

        feat = torch.cat([feat_3, feat_2], 1)   # fuse other features? cjh
        feat = self.down_query(feat)

        #   Support Feature
        feat_list = []

        if self.shot > 1:
            supp_feat = feat_list[0]
            for i in range(1, len(feat_list)):
                supp_feat += feat_list[i]
            feat /= len(feat_list)

        out_list = []
        pyramid_feat_list = []

        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin = int(feat.shape[2] * tmp_bin)
                feat_bin = nn.AdaptiveAvgPool2d(bin)(feat)
            else:
                bin = tmp_bin
                feat_bin = self.avgpool_list[idx](feat)

            merge_feat_bin = torch.cat([feat_bin], 1)
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx - 1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx - 1](rec_feat_bin) + merge_feat_bin

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(h, w),
                                           mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)

        query_feat = torch.cat(pyramid_feat_list, 1)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat
        # out = self.cls(query_feat)

        #   Output Part
        # if self.zoom_factor != 1:
        out = F.interpolate(query_feat, size=(h, w), mode='bilinear', align_corners=True)

        return out


class encoder_fuse2(nn.Module):

    def __init__(self, layers = 50, classes = 6, \
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
                 pretrained=True, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8],
                 ):
        super(encoder_fuse2, self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm
        self.criterion = criterion
        self.shot = shot
        self.ppm_scales = ppm_scales

        models.BatchNorm = BatchNorm

        print('INFO: Using ResNet {}'.format(layers))
        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                    resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256

        fea_dim = 1024 + 512

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.pyramid_bins = ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )

        factor = 1
        mask_add_num = 1
        self.init_merge = []
        self.beta_conv = []
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))

        self.init_merge = nn.ModuleList(self.init_merge)
        self.beta_conv = nn.ModuleList(self.beta_conv)

        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        x_size = x.size()
        h = x_size[2]
        w = x_size[3]

        #   Query Feature
        with torch.no_grad():
            feat_0 = self.layer0(x)
            feat_1 = self.layer1(feat_0)
            feat_2 = self.layer2(feat_1)
            feat_3 = self.layer3(feat_2)
            feat_4 = self.layer4(feat_3)

        feat = torch.cat([feat_3, feat_2], 1)   # fuse other features? cjh
        feat = self.down_query(feat)

        feat = self.res1(feat)
        feat = self.res2(feat) + feat

        out = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)

        return out

class encoder_fuse3(nn.Module):

    def __init__(self, layers = 50, classes = 6, \
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
                 pretrained=True, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8],
                 ):
        super(encoder_fuse3, self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm
        self.criterion = criterion
        self.shot = shot
        self.ppm_scales = ppm_scales

        models.BatchNorm = BatchNorm

        print('INFO: Using ResNet {}'.format(layers))
        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                    resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer2.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256

        fea_dim = 2048 + 1024 + 512

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.pyramid_bins = ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )

        factor = 1
        mask_add_num = 1
        self.init_merge = []
        self.beta_conv = []
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))

        self.init_merge = nn.ModuleList(self.init_merge)
        self.beta_conv = nn.ModuleList(self.beta_conv)

    def forward(self, x):
        x_size = x.size()
        h = x_size[2]
        w = x_size[3]

        #   Query Feature
        with torch.no_grad():
            feat_0 = self.layer0(x)
            feat_1 = self.layer1(feat_0)
            feat_2 = self.layer2(feat_1)
            feat_3 = self.layer3(feat_2)
            feat_4 = self.layer4(feat_3)

        feat = torch.cat([feat_4, feat_3, feat_2], 1)   # fuse other features? cjh
        feat = self.down_query(feat)

        out = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)

        return out

class encoder_fuse4(nn.Module):

    def __init__(self, layers = 50, classes = 6, \
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
                 pretrained=True, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8],
                 ):
        super(encoder_fuse4, self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm
        self.criterion = criterion
        self.shot = shot
        self.ppm_scales = ppm_scales

        models.BatchNorm = BatchNorm

        print('INFO: Using ResNet {}'.format(layers))
        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                    resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer2.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256

        fea_dim = 1024 + 512

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.pyramid_bins = ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )

        factor = 1
        mask_add_num = 1
        self.init_merge = []
        self.beta_conv = []
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))

        self.init_merge = nn.ModuleList(self.init_merge)
        self.beta_conv = nn.ModuleList(self.beta_conv)

    def forward(self, x):
        x_size = x.size()
        h = x_size[2]
        w = x_size[3]

        #   Query Feature
        with torch.no_grad():
            feat_0 = self.layer0(x)
            feat_1 = self.layer1(feat_0)
            feat_2 = self.layer2(feat_1)
            feat_3 = self.layer3(feat_2)
            feat_4 = self.layer4(feat_3)

        feat = torch.cat([feat_3, feat_2], 1)   # fuse other features? cjh
        feat = self.down_query(feat)

        out = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)

        return out

class encoder_fuse5(nn.Module):

    def __init__(self, layers = 50, classes = 6, \
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
                 pretrained=True, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8],
                 ):
        super(encoder_fuse5, self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm
        self.criterion = criterion
        self.shot = shot
        self.ppm_scales = ppm_scales

        models.BatchNorm = BatchNorm

        print('INFO: Using ResNet {}'.format(layers))
        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                    resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer2.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256

        fea_dim = 2048

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.pyramid_bins = ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )

        factor = 1
        mask_add_num = 1
        self.init_merge = []
        self.beta_conv = []
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))

        self.init_merge = nn.ModuleList(self.init_merge)
        self.beta_conv = nn.ModuleList(self.beta_conv)

    def forward(self, x):
        x_size = x.size()
        h = x_size[2]
        w = x_size[3]

        #   Query Feature
        with torch.no_grad():
            feat_0 = self.layer0(x)
            feat_1 = self.layer1(feat_0)
            feat_2 = self.layer2(feat_1)
            feat_3 = self.layer3(feat_2)
            feat_4 = self.layer4(feat_3)

        feat = self.down_query(feat_4)

        out = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)

        return out


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out

class encoder_fuse6(nn.Module):

    def __init__(self, layers = 50, classes = 6, \
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
                 pretrained=True, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8],
                 ):
        super(encoder_fuse6, self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm
        self.criterion = criterion
        self.shot = shot
        self.ppm_scales = ppm_scales

        models.BatchNorm = BatchNorm

        print('INFO: Using ResNet {}'.format(layers))
        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                    resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer2.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256

        fea_dim = 2048

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.pyramid_bins = ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )

        factor = 1
        mask_add_num = 1
        self.init_merge = []
        self.beta_conv = []
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))

        self.init_merge = nn.ModuleList(self.init_merge)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.decode = Classifier_Module(1024+512, [6, 12, 18, 24], [6, 12, 18, 24], 32)

    def forward(self, x):
        x_size = x.size()
        h = x_size[2]
        w = x_size[3]

        #   Query Feature
        with torch.no_grad():
            feat_0 = self.layer0(x)
            feat_1 = self.layer1(feat_0)
            feat_2 = self.layer2(feat_1)
            feat_3 = self.layer3(feat_2)
            feat_4 = self.layer4(feat_3)

        feat = torch.cat([feat_3, feat_2], 1)   # fuse other features? cjh
        feat = self.decode(feat)


        out = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)

        return out

class encoder_fuse7(nn.Module):

    def __init__(self, layers = 50, classes = 6, \
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
                 pretrained=True, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8],
                 ):
        super(encoder_fuse7, self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm
        self.criterion = criterion
        self.shot = shot
        self.ppm_scales = ppm_scales

        models.BatchNorm = BatchNorm

        print('INFO: Using ResNet {}'.format(layers))
        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                    resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4


        reduce_dim = 256

        fea_dim = 2048

        self.decode = Classifier_Module(1024+512, [6, 12, 18, 24], [6, 12, 18, 24], 32)

    def forward(self, x):
        x_size = x.size()
        h = x_size[2]
        w = x_size[3]

        #   Query Feature
        with torch.no_grad():
            feat_0 = self.layer0(x)
            feat_1 = self.layer1(feat_0)
            feat_2 = self.layer2(feat_1)
            feat_3 = self.layer3(feat_2)
            feat_4 = self.layer4(feat_3)

        feat = torch.cat([feat_3, feat_2], 1)   # fuse other features? cjh
        feat = self.decode(feat)

        out = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)

        return out

from collections import OrderedDict
from ResNetBackbone import resnet50 as Encoder
INIT_PATH = './pretrained_model/resnet50-19c8e357.pth'
class encoder_fuse8(nn.Module):

    def __init__(self, layers = 50, classes = 6, \
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
                 pretrained_path=INIT_PATH, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8],
                 ):
        super(encoder_fuse8, self).__init__()
        self.encoder = Encoder(init_path=pretrained_path)
        self.decoder = Classifier_Module(1024+512, [6, 12, 18, 24], [6, 12, 18, 24], 32)
        print('encoder_fuse8:',encoder_fuse8)

        # self.encoder = nn.Sequential(OrderedDict([
        #     ('backbone', Encoder(init_path=pretrained_path)), ('decoder', Decoder)]))

    def forward(self, x):
        x_size = x.size()
        h = x_size[2]
        w = x_size[3]

        #   Query Feature
        # with torch.no_grad():
        feat_1, feat_2, feat_3, feat_4 = self.encoder(x)


        feat = torch.cat([feat_3, feat_2], 1)   # fuse other features? cjh
        feat = self.decoder(feat)

        out = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)

        return out

class encoder_fuse9(nn.Module):

    def __init__(self, layers = 50, classes = 6, \
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
                 pretrained_path=INIT_PATH, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8],
                 ):
        super(encoder_fuse9, self).__init__()
        self.encoder = Encoder(init_path=pretrained_path)
        self.decoder = Classifier_Module(256, [6, 12, 18, 24], [6, 12, 18, 24], 32)
        print('encoder_fuse8:',encoder_fuse8)

        # self.encoder = nn.Sequential(OrderedDict([
        #     ('backbone', Encoder(init_path=pretrained_path)), ('decoder', Decoder)]))

    def forward(self, x):
        x_size = x.size()
        h = x_size[2]
        w = x_size[3]

        #   Query Feature
        # with torch.no_grad():
        feat_1, feat_2, feat_3, feat_4 = self.encoder(x)


        feat = self.decoder(feat_1)

        out = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)

        return out


class encoder_fuse10(nn.Module):
    def __init__(self, layers = 18,classes = 6, \
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
                 pretrained_path=INIT_PATH, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8],
                 ):
        super(encoder_fuse10, self).__init__()
        self.encoder = ResNet18()

    def forward(self, x):
        out = self.encoder(x)
        return out

class encoder_fuse11(nn.Module):
    def __init__(self, layers = 18,classes = 6, \
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
                 pretrained_path=INIT_PATH, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8],
                 ):
        super(encoder_fuse11, self).__init__()
        self.encoder = ResNet10()

    def forward(self, x):
        out = self.encoder(x)
        return out

class encoder_fuse12(nn.Module):

    def __init__(self, layers = 50, classes = 6, \
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
                 pretrained_path=INIT_PATH, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8],
                 ):
        super(encoder_fuse12, self).__init__()
        self.encoder = Encoder(init_path=pretrained_path)
        self.decoder = Classifier_Module(256+512, [6, 12, 18, 24], [6, 12, 18, 24], 32)
        print('encoder_fuse8:',encoder_fuse12)

        # self.encoder = nn.Sequential(OrderedDict([
        #     ('backbone', Encoder(init_path=pretrained_path)), ('decoder', Decoder)]))

    def forward(self, x):
        x_size = x.size()
        h = x_size[2]
        w = x_size[3]

        #   Query Feature
        # with torch.no_grad():
        feat_1, feat_2, feat_3, feat_4 = self.encoder(x)

        feat_2_copy = F.interpolate(feat_2, size=(feat_1.shape[2], feat_1.shape[3]), mode='bilinear',align_corners=True)
        feat = torch.cat([feat_2_copy, feat_1], 1)   # fuse other features? cjh
        feat = self.decoder(feat)

        out = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)

        return out

class encoder_fuse13(nn.Module):

    def __init__(self, layers = 50, classes = 6, \
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
                 pretrained_path=INIT_PATH, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8],
                 ):
        super(encoder_fuse13, self).__init__()
        self.encoder = Encoder(init_path=pretrained_path)
        self.decoder = Classifier_Module(256, [6, 12, 18, 24], [6, 12, 18, 24], 32)
        print('encoder_fuse8:',encoder_fuse12)

        # self.encoder = nn.Sequential(OrderedDict([
        #     ('backbone', Encoder(init_path=pretrained_path)), ('decoder', Decoder)]))

    def forward(self, x):
        x_size = x.size()
        h = x_size[2]
        w = x_size[3]

        #   Query Feature
        # with torch.no_grad():
        feat_1, feat_2, feat_3, feat_4 = self.encoder(x)

        # feat_2_copy = F.interpolate(feat_2, size=(feat_1.shape[2], feat_1.shape[3]), mode='bilinear',align_corners=True)
        # feat = torch.cat([feat_2_copy, feat_1], 1)   # fuse other features? cjh
        feat = self.decoder(feat_1)

        out = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)

        return out