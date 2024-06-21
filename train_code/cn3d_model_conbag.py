 #!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
from turtle import forward
from utils_my import group_points_2_3DV, group_points_3DV, group_points_3DV_nums
import torchvision
import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import open_clip
from typing import Tuple, Union, Callable, Optional
from collections import OrderedDict
import stgcn as SK_MODELL
# from skeleton_dataset import Feeder_dual
nstates_plus_1 = [64, 64, 256]
nstates_plus_2 = [128, 128, 256]
nstates_plus_3 = [256, 512, 1024, 1024, 1024]

vlad_dim_out = 128 * 8



class PointNet_Plus(nn.Module):
    def __init__(self, opt, num_clusters=64, gost=3, dim=512, K1 = 64, K2 = 64, suple = False):
        super(PointNet_Plus, self).__init__()
        self.temperal_num = opt.temperal_num
        self.knn_K = opt.knn_K
        self.ball_radius2 = opt.ball_radius2
        self.sample_num_level1 = opt.sample_num_level1
        self.sample_num_level2 = opt.sample_num_level2
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM  # x,y,x,c : 4
        self.num_outputs = opt.Num_Class
        self.batch = opt.batchSize
        self.dim = dim
        self.num_clusters = num_clusters
        self.gost = gost
        self.suple = suple

        # self.normalize_input = normalize_input
        self.pooling = opt.pooling

        if self.pooling == 'concatenation':
            self.dim_out = 1024
            

        self.net3DV_1 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(gost, nstates_plus_1[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[0], nstates_plus_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[1], nstates_plus_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1, K1), stride=1)
        )


        self.net3DV_3 = nn.Sequential(
            # B*259*sample_num_level2*1
            nn.Conv2d(3 + nstates_plus_2[2], nstates_plus_3[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[0]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[0], nstates_plus_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[1]),
            nn.ReLU(inplace=True),
            # B*512*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[1], nstates_plus_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[2]),
            nn.ReLU(inplace=True),
            # B*1024*sample_num_level2*1
            # nn.MaxPool2d((self.sample_num_level2, 1), stride=1)
            # B*1024*1*1
        )
        

        self.my_max_pool = nn.Sequential(nn.MaxPool2d((K2, 1), stride=1))
        # self.gobaol_max_pool = nn.Sequential(nn.MaxPool2d((self.sample_num_level2*self.gost, 1), stride=1))
        self.netR_FC = nn.Sequential(
            nn.Linear(self.dim_out, nstates_plus_3[4]),
            nn.BatchNorm1d(nstates_plus_3[4]),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(nstates_plus_3[4], self.dim),
        )
   

        self.mapping = nn.Linear(self.dim, self.num_clusters, bias=False)  # mapping weight is the properties
        self.fc = nn.Linear(1024, 120)

    def forward(self, xt, yt, if_semi= 10):

        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*_*3*sample_num_level1*1
        ###----motion stream--------
        B, d, N, k = xt.shape
        xt_3dv1 = self.net3DV_1(xt)
        xt = torch.cat((yt, xt_3dv1), 1)#.squeeze(-1)
        xt_local = self.net3DV_3(xt)  # (gost * batch) * 1024 * 64 *1
        xt = self.my_max_pool(xt_local).squeeze(-1).squeeze(-1)
        x = self.netR_FC(xt)  # (2*batch) * 128

        return x#, 0, 0, 0





class Feature_caption(nn.Module):
    def __init__(self, opt, num_class = 120):
        super(Feature_caption, self).__init__()
        

        self.encoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(512, 256),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
        )

    
    def forward(self, img):
        
        x = self.encoder(img)
        feature = self.decoder(x)
        return feature



class Backbone(nn.Module):
    def __init__(self, opt, num_class = 120):
        super(Backbone, self).__init__()
        temp_encoder  = torchvision.models.resnet18(pretrained=True)
        self.base_encoder = nn.Sequential(*list(temp_encoder.children())[:-1]) # strips off last linear layer
        self.fc = nn.Linear(512, num_class, bias=False)
        self.dim = 512

        self.netR_FC = nn.Sequential(
            nn.Linear(512, nstates_plus_3[4]),
            nn.BatchNorm1d(nstates_plus_3[4]),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(nstates_plus_3[4], self.dim),
        )

        self.channel_projection = self.net3DV_1 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(2, 3, kernel_size=(1, 1)),
            nn.BatchNorm2d(3))
    
    def forward(self, img):
        # img = self.channel_projection(img)
        x = self.base_encoder(img).squeeze().squeeze()
        feature = self.netR_FC(x)
        return feature



    
    
    
class Backbone_point_text_depth_app(nn.Module):
    def __init__(self, opt, num_class = 120, K1 = 64, K2 = 16):
        super(Backbone_point_text_depth_app, self).__init__()
        self.sup = opt.if_sup
        self.exf = opt.exf
        self.dim = 512
        
        
        if self.exf[0] == '1':
            self.point_encoder = PointNet_Plus_Temporal(opt, K1 = K1, K2 = K2, suple = False)    
        else:
            self.point_encoder = 0

        
        
        if self.exf[1] == '1':
            open_ai_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
            self.text_encoder = open_ai_model
            self.netR_FC = nn.Sequential(
                nn.Linear(512, nstates_plus_3[4]),
                nn.BatchNorm1d(nstates_plus_3[4]),
                nn.ReLU(inplace=True),
                # B*512
                nn.Linear(nstates_plus_3[4], self.dim),
            )
        else:
            self.text_encoder = 0
            self.netR_FC = 0
            
        
        if self.exf[2] == '1':
            image_encoder  = torchvision.models.resnet18(pretrained=True)
            self.image_encoder = nn.Sequential(*list(image_encoder.children())[:-1]) # strips off last linear layer
            self.netR_FC2 = nn.Sequential(
                nn.Linear(512, nstates_plus_3[4]),
                nn.BatchNorm1d(nstates_plus_3[4]),
                nn.ReLU(inplace=True),
                # B*512
                nn.Linear(nstates_plus_3[4], self.dim),
            )
        
        else:
            self.image_encoder = 0
            self.netR_FC2 = 0
        
        
        if self.exf[3] == '1':
            self.appreance_encoder = PointNet_Plus(opt, gost = 3, K1 = 64, K2 = 64, suple = False)    
        else:
            self.appreance_encoder = 0
        
        
        self.fc = nn.Linear(1024, num_class)

    
    def forward(self, image, text, xt, yt, xt_app, yt_app, modal = '1000', if_test = False):      
        
        if modal[0] == '1':
            feature_pot = self.point_encoder(xt, yt, if_test = if_test)  # x, x_single
        else:
            feature_pot = [0, 0] 
        
        
        if modal[1] == '1':
            feature_text = self.text_encoder.encode_text(text).reshape(-1, 512)
            feature_text = self.netR_FC(feature_text)
        else:
            feature_text = 0
            
        
        if modal[2] == '1':
            feature_image = self.image_encoder(image).squeeze(-1).squeeze(-1)
            feature_image = self.netR_FC2(feature_image)
        else:
            feature_image = 0
            
        if modal[3] == '1':
            feature_app = self.appreance_encoder(xt_app, yt_app)  # x, x_single
        else:
            feature_app = 0
        
        
    
        if if_test:
            # x, x_single  = feature_pot
            
            # x = F.normalize(x, p=2, dim=1)
            # x = x.reshape(2, -1, 512).permute(1, 0, 2).reshape(-1, 1024)
            
            # x_single = F.normalize(x_single, p=2, dim=1)
            # x_single = x_single.reshape(2*4, -1, 512).permute(1, 0, 2).reshape(-1, 512*8)
            
            # feature_text = F.normalize(feature_text, p=2, dim=1)
            # feature_text = feature_text.reshape(2, -1, 512).permute(1, 0, 2).reshape(-1, 1024)
            
            feature_app = F.normalize(feature_app, p=2, dim=1)
            feature_app = feature_app.reshape(2, -1, 512).permute(1, 0, 2).reshape(-1, 1024)
            
            feature_image = F.normalize(feature_image, p=2, dim=1)
            feature_image = feature_image.reshape(2, -1, 512).permute(1, 0, 2).reshape(-1, 1024)
            
            x_concate = torch.cat((feature_image, feature_app), -1)
            
            presdiction = self.fc(x_concate)
            return presdiction
        
        return feature_pot, feature_text, feature_image, feature_app


    
class Backbone_point_text_depth(nn.Module):
    def __init__(self, opt, num_class = 120, K1 = 64, K2 = 16):
        super(Backbone_point_text_depth, self).__init__()
        self.sup = opt.if_sup
        self.exf = opt.exf
        self.dim = 512
        self.tta = opt.tta
        
        
        if self.exf[0] == '1':
            self.point_encoder = PointNet_Plus_Temporal(opt, K1 = K1, K2 = K2, suple = False)    
        else:
            self.point_encoder = 0

        
        
        if self.exf[1] == '1':
            open_ai_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
            self.text_encoder = open_ai_model
            self.netR_FC = nn.Sequential(
                nn.Linear(512, nstates_plus_3[4]),
                nn.BatchNorm1d(nstates_plus_3[4]),
                nn.ReLU(inplace=True),
                # B*512
                nn.Linear(nstates_plus_3[4], self.dim),
            )
        else:
            self.text_encoder = 0
            self.netR_FC = 0
            
        
        if self.exf[2] == '1':
            image_encoder  = torchvision.models.resnet18(pretrained=True)
            self.image_encoder = nn.Sequential(*list(image_encoder.children())[:-1]) # strips off last linear layer
            self.netR_FC2 = nn.Sequential(
                nn.Linear(512, nstates_plus_3[4]),
                nn.BatchNorm1d(nstates_plus_3[4]),
                nn.ReLU(inplace=True),
                # B*512
                nn.Linear(nstates_plus_3[4], self.dim),
            )
        
        else:
            self.image_encoder = 0
            self.netR_FC2 = 0
        
        self.fc = nn.Linear(1024, num_class)

    
    def forward(self, image, text, xt, yt, modal = '100', if_test = False):      
        
        if modal[0] == '1':
            if self.tta:
                feature_pot = self.point_encoder(xt[0], yt[0], if_test = if_test)  # x, x_single
                feature_pot2 = self.point_encoder(xt[1], yt[1], if_test = if_test)  # x, x_single
            else:
                feature_pot = self.point_encoder(xt, yt, if_test = if_test)  # x, x_single
            
        else:
            feature_pot = [0, 0] 
        
        
        if modal[1] == '1':
            feature_text = self.text_encoder.encode_text(text).reshape(-1, 512)
            feature_text = self.netR_FC(feature_text)
        else:
            feature_text = 0
            
        
        if modal[2] == '1':
            if self.tta:
                feature_image = self.image_encoder(image[0]).squeeze(-1).squeeze(-1)
                feature_image = self.netR_FC2(feature_image)
                
                feature_image2 = self.image_encoder(image[1]).squeeze(-1).squeeze(-1)
                feature_image2 = self.netR_FC2(feature_image2)
            else:
                feature_image = self.image_encoder(image).squeeze(-1).squeeze(-1)
                feature_image = self.netR_FC2(feature_image)
        else:
            feature_image = 0
        
        
    
        if if_test:
            if self.tta:
                x, x_single  = feature_pot
                
                x = F.normalize(x, p=2, dim=1)
                x = x.reshape(2, -1, 512).permute(1, 0, 2).reshape(-1, 1024)
                
                x_single = F.normalize(x_single, p=2, dim=1)
                x_single = x_single.reshape(2*4, -1, 512).permute(1, 0, 2).reshape(-1, 512*8)
                
                feature_image = F.normalize(feature_image, p=2, dim=1)
                feature_image = feature_image.reshape(2, -1, 512).permute(1, 0, 2).reshape(-1, 1024)
                
                
                
                
                x2, x_single2  = feature_pot2
                
                x2 = F.normalize(x2, p=2, dim=1)
                x2 = x2.reshape(2, -1, 512).permute(1, 0, 2).reshape(-1, 1024)
                
                x_single2 = F.normalize(x_single2, p=2, dim=1)
                x_single2 = x_single2.reshape(2*4, -1, 512).permute(1, 0, 2).reshape(-1, 512*8)
                
                feature_image2 = F.normalize(feature_image2, p=2, dim=1)
                feature_image2 = feature_image2.reshape(2, -1, 512).permute(1, 0, 2).reshape(-1, 1024)
                
                
                x_concate = torch.cat((x, x_single, feature_image, x2, x_single2, feature_image2), -1)
                
                presdiction = self.fc(x_concate)
                return presdiction
            else:
                x, x_single  = feature_pot
                
                x = F.normalize(x, p=2, dim=1)
                x = x.reshape(2, -1, 512).permute(1, 0, 2).reshape(-1, 1024)
                
                x_single = F.normalize(x_single, p=2, dim=1)
                x_single = x_single.reshape(2*4, -1, 512).permute(1, 0, 2).reshape(-1, 512*8)
                
                # feature_text = F.normalize(feature_text, p=2, dim=1)
                # feature_text = feature_text.reshape(2, -1, 512).permute(1, 0, 2).reshape(-1, 1024)
                
                feature_image = F.normalize(feature_image, p=2, dim=1)
                feature_image = feature_image.reshape(2, -1, 512).permute(1, 0, 2).reshape(-1, 1024)
                
                x_concate = torch.cat((x, x_single, feature_image), -1)
                
                presdiction = self.fc(x_concate)
                return presdiction
        
        return feature_pot, feature_text, feature_image




class PointNet_Plus_Temporal(nn.Module):
    def __init__(self, opt, num_clusters=64, gost=2, dim=512, K1 = 64, K2 = 64, suple = False):
        super(PointNet_Plus_Temporal, self).__init__()
        self.temperal_num = opt.temperal_num
        self.knn_K = opt.knn_K
        self.ball_radius2 = opt.ball_radius2
        self.sample_num_level1 = opt.sample_num_level1
        self.sample_num_level2 = opt.sample_num_level2
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM  # x,y,x,c : 4
        self.num_outputs = opt.Num_Class
        self.batch = opt.batchSize
        self.dim = dim
        self.num_clusters = num_clusters
        self.gost = gost
        self.suple = suple
        self.K2 = K2

        # self.normalize_input = normalize_input
        self.pooling = opt.pooling

        if self.pooling == 'concatenation':
            self.dim_out = 1024
            

        self.net3DV_1 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(self.INPUT_FEATURE_NUM, nstates_plus_1[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[0], nstates_plus_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[1], nstates_plus_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1, K1), stride=1)
        )


        self.net3DV_3 = nn.Sequential(
            # B*259*sample_num_level2*1
            nn.Conv2d(3 + nstates_plus_2[2], nstates_plus_3[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[0]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[0], nstates_plus_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[1]),
            nn.ReLU(inplace=True),
            # B*512*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[1], nstates_plus_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[2]),
            nn.ReLU(inplace=True),
            # B*1024*sample_num_level2*1
            # nn.MaxPool2d((self.sample_num_level2, 1), stride=1)
            # B*1024*1*1
        )
        


        self.net3DV_3_t = nn.Sequential(
            # B*259*sample_num_level2*1
            nn.Conv2d(3 + nstates_plus_2[2], nstates_plus_3[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[0]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[0], nstates_plus_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[1]),
            nn.ReLU(inplace=True),
            # B*512*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[1], nstates_plus_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[2]),
            nn.ReLU(inplace=True),
            # B*1024*sample_num_level2*1
            # nn.MaxPool2d((self.sample_num_level2, 1), stride=1)
            # B*1024*1*1
        )
        
        
        

        self.my_max_pool = nn.Sequential(nn.MaxPool2d((self.K2, 1), stride=1))
        self.my_max_pool_t = nn.Sequential(nn.MaxPool2d((4, 1), stride=1))
        # self.my_max_pool_s = nn.Sequential(nn.MaxPool2d((self.K2*2, 1), stride=1))
        # self.gobaol_max_pool = nn.Sequential(nn.MaxPool2d((self.sample_num_level2*self.gost, 1), stride=1))
        self.netR_FC = nn.Sequential(
            nn.Linear(self.dim_out, nstates_plus_3[4]),
            nn.BatchNorm1d(nstates_plus_3[4]),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(nstates_plus_3[4], self.dim),
        )
        
        self.netR_FC2 = nn.Sequential(
            nn.Linear(self.dim_out, nstates_plus_3[4]),
            nn.BatchNorm1d(nstates_plus_3[4]),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(nstates_plus_3[4], self.dim),
        )
        
   

        self.mapping = nn.Linear(self.dim, self.num_clusters, bias=False)  # mapping weight is the properties
        self.fc = nn.Linear(1024, 120)

    def forward(self, xt, yt, if_semi= 10, if_test= False):

        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*_*3*sample_num_level1*1
        ###----motion stream--------

        B, d, N, k = xt.shape
        xt_3dv1 = self.net3DV_1(xt)
        xt = torch.cat((yt, xt_3dv1), 1)#.squeeze(-1)
        
        xt = xt.reshape(2, 5, -1, 3 + nstates_plus_2[2], self.K2, 1)
        x_whole = xt[:, 0, ...].reshape(-1, 3 + nstates_plus_2[2], self.K2, 1)
        x_single = xt[:, 1:, ...].reshape(-1, 3 + nstates_plus_2[2], self.K2, 1)
        
        xt_local = self.net3DV_3(x_whole)  # (gost * batch) * 1024 * 64 *1
        x_whole = self.my_max_pool(xt_local).squeeze(-1).squeeze(-1)    
        
        x_single = self.net3DV_3_t(x_single)  # (gost * batch) * 1024 * 64 *1
        x_single = self.my_max_pool(x_single).squeeze(-1).squeeze(-1)    
        x_single = x_single.reshape(2, 4, -1, 1024, 1, 1).permute(0, 2, 3, 1, 4, 5)
        x_single_all = x_single.reshape(-1, 1024, 4, 1)
        x_single = self.my_max_pool_t(x_single_all).squeeze(-1).squeeze(-1)
        
        x = self.netR_FC(x_whole)  # (2*batch) * 128
        
        if if_test:
            # **************test 
            x_single_all = x_single_all.permute(2, 0, 1, 3).reshape(-1, 1024) # (4*2*batch) * 1024
            x_single = self.netR_FC2(x_single_all)
        else:
            x_single = self.netR_FC2(x_single)
        
        
        
        
        
        if self.suple:
            disribution,_ = torch.sort(xt_local.squeeze(), dim = -1, descending=True)
            # disribution = disribution[:, :, np.arange(0, 20, 5)].permute(2, 0 ,1).reshape(-1, xt_local.shape[1])
            disribution = disribution[:, :, :32].permute(2, 0 ,1).reshape(-1, xt_local.shape[1])
            suplement = self.netR_FC(disribution)
            return suplement
        if if_semi==0:
            feature_pot = x.reshape(2, -1, 512).permute(1, 0 ,2).reshape(-1, 1024)
            x_nor = F.normalize(feature_pot, p=2, dim=1)
            presdiction = self.fc(x_nor)
            return presdiction
        elif if_semi == 1:
            disribution = xt_local.squeeze().permute(2, 0 ,1).reshape(-1, xt_local.shape[1])
            suplement = self.netR_FC(disribution)
            return suplement

        elif if_semi == 3:
            x = F.normalize(x, p=2, dim=1)
            x = x.reshape(2, -1, 512).permute(1, 0, 2).reshape(-1, 1024)
            x = self.fc(x)
            return x
        
        elif if_semi == 4:
            x = F.normalize(x, p=2, dim=1)
            x = x.reshape(2, -1, 512).permute(1, 0, 2).reshape(-1, 1024)
            
            x_single = F.normalize(x_single, p=2, dim=1)
            x_single = x_single.reshape(2, -1, 512).permute(1, 0, 2).reshape(-1, 1024)
            
            x_concate = torch.cat((x, x_single), -1)
            
            x = self.fc(x_concate)
            return x
        
        elif if_semi == 5:
            x = F.normalize(x, p=2, dim=1)
            x = x.reshape(2, -1, 512).permute(1, 0, 2).reshape(-1, 1024)
            
            x_single = F.normalize(x_single, p=2, dim=1)
            x_single = x_single.reshape(2, -1, 512).permute(1, 0, 2).reshape(-1, 1024)
            
            x_concate = torch.cat((x, x_single), -1)
            
            return x_concate

        return x, x_single #, 0, 0, 0





class PointNet_Plus(nn.Module):
    def __init__(self, opt, num_clusters=64, gost=4, dim=512, K1 = 64, K2 = 64, suple = False):
        super(PointNet_Plus, self).__init__()
        self.temperal_num = opt.temperal_num
        self.knn_K = opt.knn_K
        self.ball_radius2 = opt.ball_radius2
        self.sample_num_level1 = opt.sample_num_level1
        self.sample_num_level2 = opt.sample_num_level2
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM  # x,y,x,c : 4
        self.num_outputs = opt.Num_Class
        self.batch = opt.batchSize
        self.dim = dim
        self.num_clusters = num_clusters
        self.gost = gost
        self.suple = suple

        # self.normalize_input = normalize_input
        self.pooling = opt.pooling

        if self.pooling == 'concatenation':
            self.dim_out = 1024
            

        self.net3DV_1 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(gost, nstates_plus_1[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[0], nstates_plus_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[1], nstates_plus_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1, K1), stride=1)
        )


        self.net3DV_3 = nn.Sequential(
            # B*259*sample_num_level2*1
            nn.Conv2d(3 + nstates_plus_2[2], nstates_plus_3[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[0]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[0], nstates_plus_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[1]),
            nn.ReLU(inplace=True),
            # B*512*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[1], nstates_plus_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[2]),
            nn.ReLU(inplace=True),
            # B*1024*sample_num_level2*1
            # nn.MaxPool2d((self.sample_num_level2, 1), stride=1)
            # B*1024*1*1
        )
        

        self.my_max_pool = nn.Sequential(nn.MaxPool2d((K2, 1), stride=1))
        # self.gobaol_max_pool = nn.Sequential(nn.MaxPool2d((self.sample_num_level2*self.gost, 1), stride=1))
        self.netR_FC = nn.Sequential(
            nn.Linear(self.dim_out, nstates_plus_3[4]),
            nn.BatchNorm1d(nstates_plus_3[4]),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(nstates_plus_3[4], self.dim),
        )
   

        self.mapping = nn.Linear(self.dim, self.num_clusters, bias=False)  # mapping weight is the properties
        self.fc = nn.Linear(1024, 120)

    def forward(self, xt, yt, if_semi= 10):

        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*_*3*sample_num_level1*1
        ###----motion stream--------
        B, d, N, k = xt.shape
        xt_3dv1 = self.net3DV_1(xt)
        xt = torch.cat((yt, xt_3dv1), 1)#.squeeze(-1)
        xt_local = self.net3DV_3(xt)  # (gost * batch) * 1024 * 64 *1
        
        
        xt = self.my_max_pool(xt_local).squeeze(-1).squeeze(-1)
        x = self.netR_FC(xt)  # (2*batch) * 128
        
        if self.suple:
            disribution,_ = torch.sort(xt_local.squeeze(), dim = -1, descending=True)
            # disribution = disribution[:, :, np.arange(0, 20, 5)].permute(2, 0 ,1).reshape(-1, xt_local.shape[1])
            disribution = disribution[:, :, :32].permute(2, 0 ,1).reshape(-1, xt_local.shape[1])
            suplement = self.netR_FC(disribution)
            return suplement
        if if_semi==0:
            feature_pot = x.reshape(2, -1, 512).permute(1, 0 ,2).reshape(-1, 1024)
            x_nor = F.normalize(feature_pot, p=2, dim=1)
            presdiction = self.fc(x_nor)
            return presdiction
        elif if_semi == 1:
            disribution = xt_local.squeeze().permute(2, 0 ,1).reshape(-1, xt_local.shape[1])
            suplement = self.netR_FC(disribution)
            return suplement

        elif if_semi == 3:
            x = F.normalize(x, p=2, dim=1)
            x = x.reshape(2, -1, 512).permute(1, 0, 2).reshape(-1, 1024)
            x = self.fc(x)
            return x

        elif if_semi == 4:
            x = F.normalize(x, p=2, dim=1)
            x = x.reshape(4, -1, 512).permute(1, 0, 2).reshape(-1, 1024*2)
            x = self.fc(x)
            return x

        return x#, 0, 0, 0
 
    
class Skeleton_model(nn.Module):
    def __init__(self,opt,num_clusters=8,gost=1,dim=512,normalize_input=True):
        super(Skeleton_model, self).__init__()
        # self.base_encoder = SK_MODELL
        graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
        self.base_encoder = SK_MODELL.Model(in_channels=3, hidden_channels=16,
                                                hidden_dim=256, num_class=dim,
                                                dropout=0.5, graph_args=graph_args,
                                                edge_importance_weighting=True)
        self.netR_FC = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(dim, dim),
        )
        
        self.fc = nn.Linear(dim, 120)
        
    
    def forward(self, text_token):
        feature = self.base_encoder(text_token)
        out = self.netR_FC(feature)
        
        return out
    
class Text_model(nn.Module):
    def __init__(self,opt,num_clusters=8,gost=1,dim=512,normalize_input=True):
        super(Text_model, self).__init__()
        open_ai_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
        self.text_encoder = open_ai_model
        self.netR_FC = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(512, dim),
        )
        
        self.fc = nn.Linear(512, 120)
        
    
    def forward(self, text_token):
        feature = self.text_encoder.encode_text(text_token).squeeze()
        out = self.netR_FC(feature)
        
        return out

class Text_Skeleton_model(nn.Module):
    def __init__(self,opt,num_clusters=8,gost=1,dim=512,normalize_input=True):
        super(Text_Skeleton_model, self).__init__()
        self.text_encoder = Text_model(opt)
        self.slenton_encoder = Skeleton_model(opt)
        
    def forward(self, text_token, skleton_token):
        text_feature = self.text_encoder(text_token)
        skeleton_feature = self.slenton_encoder(skleton_token)
        return text_feature, skeleton_feature
    




def distributed_sinkhorn(Q, nmb_iters):
    with torch.no_grad():
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        # dist.all_reduce(sum_Q)
        Q /= sum_Q
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (Q.shape[1])
        for it in range(nmb_iters):
            u = torch.sum(Q, dim=1)
            # dist.all_reduce(u)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor
 
 

    




