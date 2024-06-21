# -*- coding: utf-8 -*-
import logging

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
import tqdm
import shutil
import collections
import argparse
import random
import time
import numpy as np
import torch.nn.functional as F

# *******************************mode change to use dgcnn
import cn3d_model_conbag as MODELL
from cn3d_pretrained_point_dataset import NTU_RGBD_depth
# from cn3d_data_load import deal_data_new

# from utils import group_points,group_points_pro

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils_my import group_points_3DV, Info_NCE, group_points_3DV_nums

id = 2
kener = [2, 4, 5]


def main(args=None):
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=121, help='number of epochs to train for')
    parser.add_argument('--INPUT_FEATURE_NUM', type=int, default=4, help='number of input point features')
    parser.add_argument('--temperal_num', type=int, default=3, help='number of input point features')
    parser.add_argument('--pooling', type=str, default='concatenation',
                        help='how to aggregate temporal split features: vlad | concatenation | bilinear')
    parser.add_argument('--dataset', type=str, default='ntu120',
                        help='how to aggregate temporal split features: ntu120 | ntu60')

    parser.add_argument('--weight_decay', type=float, default=0.0008, help='weight decay (SGD only)')
    parser.add_argument('--learning_rate', type=float, default=0.00025, help='learning rate at t=0')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (SGD only)')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')

    parser.add_argument('--save_root_dir', type=str, default='../data/models/',
                        help='output folder')
    parser.add_argument('--model', type=str, default='', help='model name for training resume')
    parser.add_argument('--optimizer', type=str, default='', help='optimizer name for training resume')

    parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
    parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id')  # CUDA_VISIBLE_DEVICES=0 python train.py

    # defined by the DGCNN
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout rate')

    parser.add_argument('--learning_rate_decay', type=float, default=1e-7, help='learning rate decay')

    parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
    parser.add_argument('--SAMPLE_NUM', type=int, default=512, help='number of sample points')

    # class change to 2 class
    parser.add_argument('--Num_Class', type=int, default=512, help='number of outputs')

    parser.add_argument('--knn_K', type=int, default=64, help='K for knn search')
    parser.add_argument('--sample_num_level1', type=int, default=64, help='number of first layer groups')
    parser.add_argument('--sample_num_level2', type=int, default=64, help='number of second layer groups')
    parser.add_argument('--ball_radius', type=float, default=0.16,
                        help='square of radius for ball query in level 1')  # 0.025 -> 0.05 for detph
    parser.add_argument('--ball_radius2', type=float, default=0.25,
                        help='square of radius for ball query in level 2')  # 0.08 -> 0.01 for depth

    parser.add_argument('--exf', type=str,default='110',help='text or point')   # point text image
    parser.add_argument('--if_sup',default=False,help='text or point')
    parser.add_argument('--round', type=int, default=0,help='fine_tune_round') 

    opt = parser.parse_args()
    print(opt)

    torch.cuda.set_device(opt.main_gpu)

    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    try:
        os.makedirs(opt.save_root_dir)
    except OSError:
        pass

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()

    # data use 3dv point cloud generate from NTU120 to train contrast net
    # data struct is defined as :
    data_train =NTU_RGBD_depth(root_path='../data/raw/', opt=opt,
                          # opt.root_path
                          DATA_CROSS_VIEW=False,
                          full_train=True,
                          validation=False,
                          test=False,
                          Transform=False,
                          DATA_CROSS_SET= True,
                          if_sequence=True
                          )

    # get all the data fold into batches
    train_loader = DataLoader(dataset=data_train, batch_size=opt.batchSize, shuffle=True, drop_last=True, num_workers=16)
    num_crop = 2
    K2= 16
    netR = MODELL.Backbone_point_text_depth(opt, K2 = K2)
    print('loding the  mode...................')
    
    netR = torch.nn.DataParallel(netR).cuda()
    netR.cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(netR.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999), eps=1e-06)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    MODE = 1
    SAVE_MODE = 2


    print('\n MODE is :', MODE, 'save mode is : ', SAVE_MODE)

    for epoch in range(0, opt.nepoch):
        netR.train()
        loss_sigma = 0.0

        loss_mode = 1

        # need to rewite data_get
        for i, data in enumerate(tqdm(train_loader, 0)):
            token_raw, token_aug, rgb_image, out_points, v_name, label = data
            
            
            modal_choose = list(str(opt.exf))

            if modal_choose[0] =='1':
                pot = out_points[:, :, :2048, :].permute(1,0,2,3).reshape(-1, 2048, 4).type(torch.FloatTensor).cuda()    # 2*T *batch   d * c
                xt, yt = group_points_3DV_nums(pot[:, :512, :].clone(), sample_num_level1 = K2)  # batch_size, 8, 512, 64    batch_size, 3, 512, 1
            else:
                xt, yt = 0, 0
            
            if modal_choose[1] =='1':
                text = torch.cat((token_raw, token_aug), dim=0).cuda()
            else:
                text = 0
            
            if modal_choose[2] =='1':
                rgb_image = 0
            else:
                rgb_image = 0
            
            
            feature_text, feature_pot, _ = netR(text, xt, yt)
            loss = get_loss(feature_text, feature_pot, opt, criterion)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)
            torch.cuda.synchronize()
            loss_sigma += loss.item() 
            # print(loss.item(),loss1.item(), loss2.item())

        logging.info('{} --epoch{} ==Average loss:{}'.format('Valid', epoch, loss_sigma / (i + 1)))
        print('epoch:', epoch, 'loss mode is :', loss_mode, '--loss:', loss_sigma / (i + 1))
        if epoch>40:
            if epoch % 60 == 0:
                torch.save(netR.module.state_dict(), '%s/ntu_point_text_cv_%d_%d.pth' % (opt.save_root_dir, opt.round, epoch))



def get_loss(feature_text, feature_pot, opt, criterion):
    fusion_feature1 = torch.cat((feature_text[:opt.batchSize], feature_pot[:opt.batchSize]), dim = 0)
    fusion_feature2 = torch.cat((feature_text[opt.batchSize:], feature_pot[opt.batchSize:]), dim = 0)

    logits_p, labels =  Info_NCE(feature_text, opt)
    logits_p1, labels1 =  Info_NCE(feature_pot, opt)
    logits_p2, labels2 =  Info_NCE(fusion_feature1, opt)
    logits_p3, labels3 =  Info_NCE(fusion_feature2, opt)

    loss1 = criterion(logits_p, labels) 
    loss2 = criterion(logits_p1, labels1)
    loss3 = criterion(logits_p2, labels2)
    loss4 = criterion(logits_p3, labels3)
    # loss = criterion(logits_p, labels) + criterion(logits_p1, labels1) + criterion(logits_p2, labels2) + criterion(logits_p3, labels3)
    loss = loss1 +loss2 +loss3 +loss4
    # print(loss.item())
    return loss



if __name__ == '__main__':
    main()


