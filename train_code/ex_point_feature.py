# -*- coding: utf-8 -*-
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
# from transfer_feature_to_token import finding_more_text_via_point
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils_my import group_points_3DV, Info_NCE

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
    parser.add_argument('--save_path', type=str, default='../data/feature/',help='fine_tune_round') 

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
                          DATA_CROSS_SET= True
                          )
    
    data_test =NTU_RGBD_depth(root_path='../data/raw/', opt=opt,
                          # opt.root_path
                          DATA_CROSS_VIEW=False,
                          full_train=True,
                          validation=False,
                          test=True,
                          Transform=False,
                          DATA_CROSS_SET=True
                          )

    # get all the data fold into batches
    train_loader = DataLoader(dataset=data_train, batch_size=opt.batchSize, shuffle=False, drop_last=False, num_workers=16)
    test_loader = DataLoader(dataset=data_test, batch_size=opt.batchSize, shuffle=False, drop_last=False, num_workers=16)
    num_crop = 2
    netR = MODELL.PointNet_Plus(opt)
    
    # self supervised
    save_path = opt.save_path+str(opt.round).zfill(3)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_name = opt.save_root_dir + 'ntu_point_text_cv_'+opt.round+'_60.pth'  
    print('loding the  mode...................', model_name)
    netR.load_state_dict(torch.load(model_name), strict= False)
    netR = torch.nn.DataParallel(netR).cuda()
    netR.cuda()



    MODE = 1
    SAVE_MODE = 2

    print('\n MODE is :', MODE, 'save mode is : ', SAVE_MODE)

    with torch.no_grad():
        netR.eval()
        loss_sigma = 0.0

        loss_mode = 1

        # need to rewite data_get
        for i, data in enumerate(tqdm(train_loader, 0)):
            out_points, v_name, label = data
            
            # img = torch.cat((raw_img, aug_img), dim=0).type(torch.FloatTensor).cuda()
            # img = img.permute(0, 3, 1,2)

            pot = out_points[:, :, :512, :].permute(1,0,2,3).reshape(-1, 512, 4).type(torch.FloatTensor).cuda()
            opt.ball_radius = opt.ball_radius + random.uniform(-0.02, 0.02)
            xt, yt = group_points_3DV(pot, opt)  # batch_size, 8, 512, 64    batch_size, 3, 512, 1

            feature = netR(xt, yt)
            save_single_feature(feature, save_path, v_name, num_crop=64)
        
        # need to rewite data_get
        for i, data in enumerate(tqdm(test_loader, 0)):
            out_points, v_name, label = data
            
            # img = torch.cat((raw_img, aug_img), dim=0).type(torch.FloatTensor).cuda()
            # img = img.permute(0, 3, 1,2)

            pot = out_points[:, :, :512, :].permute(1,0,2,3).reshape(-1, 512, 4).type(torch.FloatTensor).cuda()
            opt.ball_radius = opt.ball_radius + random.uniform(-0.02, 0.02)
            xt, yt = group_points_3DV(pot, opt)  # batch_size, 8, 512, 64    batch_size, 3, 512, 1

            feature = netR(xt, yt)
            save_single_feature(feature, save_path, v_name, num_crop=64)
    
    finding_more_text_via_point(point_feature_path = save_path, round = opt.round)

def save_single_feature(feature, save_path, name, num_crop = 2):
    save_dim = feature.shape[-1]
    feature = feature.detach().cpu().numpy().reshape(num_crop, -1 , save_dim).transpose(1, 0, 2).reshape(-1, num_crop * save_dim)
    for batch_i in range(feature.shape[0]):
        f_p = save_path + name[batch_i] + '.npy'
        np.save(f_p, feature[batch_i])              



def finding_more_text_via_point(save_path = "../data/neibor_indexs/", point_feature_path = " last round learned point cloud feature  save path", round = 0):
    save_path = '../data/neibor_indexs/' +str(round).zfill(3)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    all_files = os.listdir(point_feature_path)
    all_files.sort()
    DIMM = 512*2
    print('totlly samples:', len(all_files))
    all_files = all_files#[:10000]
    que_feature = []
    file_name = []
    for sub_file in all_files:
        view = int(sub_file[5:8])
        if view in train_ids_camera:
            temp_feature= np.load(point_feature_path + sub_file)[:DIMM]
            que_feature.append(temp_feature)
            file_name.append(sub_file)
    que_feature = np.array(que_feature)
    que_feature = torch.from_numpy(que_feature)
    que_feature = F.normalize(que_feature, p=2, dim=1).numpy()
    
    all_files = os.listdir(point_feature_path)
    all_files.sort()
    print('totlly samples:', len(all_files))
    idd = 0
    for sub_file in all_files:
        view = int(sub_file[5:8])
        if view in train_ids_camera:
            continue
        print(sub_file, idd)
        idd +=1
        temp_feature= np.load(point_feature_path + sub_file)[:DIMM].reshape(1, -1)
        temp_feature = torch.from_numpy(temp_feature)
        temp_feature = F.normalize(temp_feature, p=2, dim=1).numpy()
        similarity = temp_feature@(que_feature.transpose(1, 0)) # 1 * K
        # indexx = np.argmax(similarity)
        indexx = np.argsort(-similarity)
        np.save(save_path+sub_file, file_name[indexx[0, 1]])
        # np.save(save_path+sub_file, indexx[0, 0:20])    # similar_text_index      similar_text_index_r. similar_text_index_r
        print(sub_file, 'crosbuding to: ', all_files[indexx[0, 1]], all_files[indexx[0, 2]], all_files[indexx[0, 3]])



if __name__ == '__main__':
    main()


