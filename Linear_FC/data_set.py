import os
import tqdm
import torch
import re
import collections
import imageio
import random

from tqdm import tqdm

from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import scipy.io as sio
import torch.nn.functional as F

fx = 365.481
fy = 365.481
cx = 257.346
cy = 210.347
# rose@ntu.edu.sg
sample_num_level1 = 512
sample_num_level2 = 128

TRAIN_IDS_60 = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
# TRAIN_IDS_60=[1, 2]

TRAIN_IDS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49,
             50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98,
             100, 103]
TRAIN_VALID_IDS = ([1, 2, 5, 8, 9, 13, 14, 15, 16, 18, 19, 27, 28, 31, 34, 38], [4, 17, 25, 35])
TRAIN_SET = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
compiled_regex = re.compile('.*S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3}).*')
SAMPLE_NUM = 2048


class LIner_NTU(Dataset):
    """NTU depth human masked datasets"""

    def __init__(self, root_path, opt,
                 full_train=True,
                 test=False,
                 validation=False,
                 DATA_CROSS_VIEW=True,
                 DATA_CROSS_SET =False,
                 Transform=False):

        self.DATA_CROSS_VIEW = DATA_CROSS_VIEW
        self.root_path = root_path
        self.test = test


        self.point_vids = os.listdir(self.root_path)
        self.point_vids.sort()

        self.TRAIN_IDS = TRAIN_IDS
        if opt.dataset == 'ntu60':
            indx = self.point_vids.index('S017C003P020R002A060.npy')# S004C001P003R001A001.npy# 'S017C003P017R002A002.npy.mat' # ('S003C001P001R001A001.npy.mat')#('S017C003P020R002A060.npy')
            self.point_vids = self.point_vids[:indx]
            # self.TRAIN_IDS = TRAIN_IDS_60

        self.num_clouds = len(self.point_vids)
        print(self.num_clouds)

        self.point_data = self.load_data()
        self.set_splits()
        self.id_to_action = list(pd.DataFrame(self.point_data)['action'] - 1)
        self.id_to_vidName = list(pd.DataFrame(self.point_data)['video_cloud_name'])

        self.train = (test == False) and (validation == False)
        if DATA_CROSS_SET ==False:       
            if DATA_CROSS_VIEW == False:
                if test:
                    self.vid_ids = self.test_split_subject.copy()
                elif validation:
                    self.vid_ids = self.validation_split_subject.copy()
                elif full_train:
                    self.vid_ids = self.train_split_subject.copy()
                else:
                    self.vid_ids = self.train_split_subject_with_validation.copy()
            else:
                if test:
                    self.vid_ids = self.test_split_camera.copy()
                else:
                    self.vid_ids = self.train_split_camera.copy()
        else:
            if test:
                self.vid_ids = self.test_split_set.copy()
            else:
                self.vid_ids = self.train_split_set.copy()

        print('num_data:', len(self.vid_ids))

    def __getitem__(self, idx):
        vid_id = self.vid_ids[idx]
        vid_name = self.id_to_vidName[vid_id]
        v_name = vid_name[:20] 
        label = self.id_to_action[vid_id]
        aug = [0,2,4,6,8]
        # feature_all = np.load(self.root_path +v_name+'.npy').reshape(-1)

        feature_all = self.read_multi_rank_maxpool(v_name)
        
    
        return feature_all, label, v_name
    
    def read_multi_rank_maxpool(self, v_name):
        feature_r = np.load('the round 2 feature save path ').reshape(66, -1)#.reshape(10,-1)
        feature_t = feature_r[64:].reshape(-1)
        feature_r = feature_r[:64].reshape(32, 2, -1)[np.arange(0,32,5)[:5], :, :].reshape(-1)
        feature_all = feature_r 
        # feature_all = np.concatenate((feature_r, feature_t), 0)
        return  feature_all


    def __len__(self):
        return len(self.vid_ids)


    def load_data(self):
        self.point_data = []
        for cloud_idx in tqdm(range(self.num_clouds), "Getting video info"):
            self.point_data.append(self.get_pointdata(cloud_idx))

        return self.point_data

    def get_pointdata(self, vid_id):

        vid_name = self.point_vids[vid_id]
        match = re.match(compiled_regex, vid_name)
        setup, camera, performer, replication, action = [*map(int, match.groups())]
        return {
            'video_cloud_name': vid_name,
            'video_index': vid_id,
            'video_set': (setup, camera),
            'setup': setup,
            'camera': camera,
            'performer': performer,
            'replication': replication,
            'action': action,
        }

    def set_splits(self):
        '''
        Sets the train/test splits
        Cross-Subject Evaluation:
            Train ids = 1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27,
                        28, 31, 34, 35, 38
        Cross-View Evaluation:
            Train camera views: 2, 3
        '''
        # Save the dataset as a dataframe
        dataset = pd.DataFrame(self.point_data)

        # Get the train split ids
        train_ids_camera = [2, 3]

        # Cross-Subject splits
        self.train_split_subject = list(
            dataset[dataset.performer.isin(self.TRAIN_IDS)]['video_index'])
        self.train_split_subject_with_validation = list(
            dataset[dataset.performer.isin(TRAIN_VALID_IDS[0])]['video_index'])
        self.validation_split_subject = list(
            dataset[dataset.performer.isin(TRAIN_VALID_IDS[1])]['video_index'])
        self.test_split_subject = list(
            dataset[~dataset.performer.isin(self.TRAIN_IDS)]['video_index'])

        # Cross-View splits
        self.train_split_camera = list(
            dataset[dataset.camera.isin(train_ids_camera)]['video_index'])
        self.test_split_camera = list(
            dataset[~dataset.camera.isin(train_ids_camera)]['video_index'])

        # Cross Set splits
        self.train_split_set = list(
            dataset[dataset.setup.isin(TRAIN_SET)]['video_index'])
        self.test_split_set = list(
            dataset[~dataset.setup.isin(TRAIN_SET)]['video_index'])


class UWA3D_liner(Dataset):
    """NTU depth human masked datasets"""

    def __init__(self, root_path, opt,
                 test=False,
                 ):

        self.root_path = root_path
        self.point_vids = os.listdir(self.root_path)
        self.point_vids.sort()

        self.TRAIN_IDS = TRAIN_IDS

        self.num_clouds = len(self.point_vids)
        print(self.num_clouds)

        self.point_data = self.load_data()
        self.set_splits()
        self.id_to_action = list(pd.DataFrame(self.point_data)['action'] - 1)
        self.id_to_vidName = list(pd.DataFrame(self.point_data)['video_cloud_name'])

        self.train = (test == False) 

        if test:
            self.vid_ids = self.test_split_camera.copy()
        else:
            self.vid_ids = self.train_split_camera.copy()


        print('num_data:', len(self.vid_ids))


    def __getitem__(self, idx):
        vid_id = self.vid_ids[idx]
        vid_name = self.id_to_vidName[vid_id]
        v_name = vid_name[:-4]
        r_path = '/data/data2/tanbo/data/data3/tanbo/ntu/ablation_study/all_features/UWA3D/r/' + v_name + '.npy'
        # x_path = '/data/data2/tanbo/data/data3/tanbo/ntu/ablation_study/all_features/UWA3D/app/' + v_name + '.npy'
        # y_path = '/data/data2/tanbo/data/data3/tanbo/ntu/ablation_study/all_features/UWA3D/y/' + v_name + '.npy'
        # z_path = '/data/data2/tanbo/data/data3/tanbo/ntu/ablation_study/all_features/UCLA/r_e/' + v_name + '.npy'
        
        
        x_path = '/data/data2/tanbo/data/data3/tanbo/ntu/ablation_study/all_features/UWA3D/x/' + v_name + '.npy'
        y_path = '/data/data2/tanbo/data/data3/tanbo/ntu/ablation_study/all_features/UWA3D/y/' + v_name + '.npy'
        z_path = '/data/data2/tanbo/data/data3/tanbo/ntu/ablation_study/all_features/UWA3D/z/' + v_name + '.npy'
        
        a_path = '/data/data2/tanbo/data/data3/tanbo/ntu/ablation_study/all_features/UWA3D/app/' + v_name + '.npy'

        feature_r = np.load(r_path)
        feature_x = np.load(x_path)
        feature_y = np.load(y_path)
        feature_z = np.load(z_path)
        feature_a = np.load(a_path)
        feature = np.concatenate((feature_r, feature_x, feature_z, feature_y, feature_a), 0)
        label = self.id_to_action[vid_id]
        return feature, label


    def __len__(self):
        return len(self.vid_ids)


    def load_data(self):
        self.point_data = []
        for cloud_idx in tqdm(range(self.num_clouds), "Getting video info"):
            self.point_data.append(self.get_pointdata(cloud_idx))

        return self.point_data

    def get_pointdata(self, vid_id):
    
        vid_name = self.point_vids[vid_id]
        match = re.match(compiled_regex, vid_name)
        # setup, camera, performer, replication, action = [*map(int, match.groups())]
        return {
            'video_cloud_name': vid_name,
            'video_index': vid_id,
            # 'video_set': (setup, camera),
            # 'setup': setup,
            'camera': int(vid_name[-6:-4]),
            # 'performer': performer,
            # 'replication': replication,
            'action': int(vid_name[1:3]),
        }
        

    def set_splits(self):
        '''
        Sets the train/test splits
        Cross-Subject Evaluation:
            Train ids = 1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27,
                        28, 31, 34, 35, 38
        Cross-View Evaluation:
            Train camera views: 2, 3
        '''
        # Save the dataset as a dataframe
        dataset = pd.DataFrame(self.point_data)

        # Get the train split ids
        train_ids_camera = [1, 2]
        test_ids_camera = [4]

        # Cross-View splits
        self.train_split_camera = list(
            dataset[dataset.camera.isin(train_ids_camera)]['video_index'])
        self.test_split_camera = list(
            dataset[dataset.camera.isin(test_ids_camera)]['video_index'])
            # dataset[~dataset.camera.isin(train_ids_camera)]['video_index'])
            
            

class UCLA_liner(Dataset):
    """NTU depth human masked datasets"""

    def __init__(self, root_path, opt,
                 test=False,
                 ):

        self.root_path = root_path
        self.point_vids = os.listdir(self.root_path)
        self.point_vids.sort()

        self.TRAIN_IDS = TRAIN_IDS

        self.num_clouds = len(self.point_vids)
        print(self.num_clouds)

        self.point_data = self.load_data()
        self.set_splits()
        self.id_to_action = list(pd.DataFrame(self.point_data)['action'] - 1)
        self.id_to_vidName = list(pd.DataFrame(self.point_data)['video_cloud_name'])

        self.train = (test == False) 

        if test:
            self.vid_ids = self.test_split_camera.copy()
        else:
            self.vid_ids = self.train_split_camera.copy()


        print('num_data:', len(self.vid_ids))
    
    
    ###************** for slow fast training ***********************
    def __getitem__(self, idx):
        vid_id = self.vid_ids[idx]
        vid_name = self.id_to_vidName[vid_id]
        v_name = vid_name[:-4]
        label = self.id_to_action[vid_id]
        # normal resolution:
        
        # feature_all = np.load('/data/data1/ntu/ICCV2023/feature/ntu60/cv/pont_detection/' +v_name+'.npy').reshape(-1)[512*2:]
        # feature_r = np.load('/data/data2/UCLA/pointnet/' +v_name+'.npy').reshape(64, -1)
        
        feature_r = np.load(self.root_path + v_name+'.npy').reshape(-1)
        # feature_r = np.load('/data/data2/UCLA/pointnet_text/' +v_name+'.npy').reshape(66, -1)
        # feature_t = feature_r[64:].reshape(-1)
        # feature_r = feature_r[:64].reshape(32, 2, -1)[np.arange(0,32,5)[:5], :, :].reshape(-1)
        
        # feature_all = np.concatenate((feature_t,feature_r), 0)
        feature_all = feature_r
    
        return feature_all.reshape(-1), label
        
    


    def __len__(self):
        return len(self.vid_ids)


    def load_data(self):
        self.point_data = []
        for cloud_idx in tqdm(range(self.num_clouds), "Getting video info"):
            self.point_data.append(self.get_pointdata(cloud_idx))

        return self.point_data

    def get_pointdata(self, vid_id):
    
        vid_name = self.point_vids[vid_id]
        match = re.match(compiled_regex, vid_name)
        # setup, camera, performer, replication, action = [*map(int, match.groups())]
        return {
            'video_cloud_name': vid_name,
            'video_index': vid_id,
            # 'video_set': (setup, camera),
            # 'setup': setup,
            'camera': int(vid_name[-6:-4]),
            # 'performer': performer,
            # 'replication': replication,
            'action': int(vid_name[1:3]),
        }
        

    def set_splits(self):
        '''
        Sets the train/test splits
        Cross-Subject Evaluation:
            Train ids = 1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27,
                        28, 31, 34, 35, 38
        Cross-View Evaluation:
            Train camera views: 2, 3
        '''
        # Save the dataset as a dataframe
        dataset = pd.DataFrame(self.point_data)

        # Get the train split ids
        train_ids_camera = [1, 2]

        # Cross-View splits
        self.train_split_camera = list(
            dataset[dataset.camera.isin(train_ids_camera)]['video_index'])
        self.test_split_camera = list(
            dataset[~dataset.camera.isin(train_ids_camera)]['video_index'])

    
class ModeNet(Dataset):
    """NTU depth human masked datasets"""

    def __init__(self, root_path, opt,
                 test=False,
                 ):

        
        self.root_path = root_path
        self.point_vids = os.listdir(self.root_path)
        self.point_vids.sort()

        self.num_clouds = len(self.point_vids)
        print(self.num_clouds)

        self.point_data = self.load_data()
        self.set_splits()
        self.id_to_action = list(pd.DataFrame(self.point_data)['action'])
        self.id_to_vidName = list(pd.DataFrame(self.point_data)['video_cloud_name'])

        self.train = (test == False) 
        if test:
            self.vid_ids = self.test_split_set.copy()
        else:
            self.vid_ids = self.train_split_set.copy()

        print('num_data:', len(self.vid_ids))


    def __getitem__(self, idx):
        vid_id = self.vid_ids[idx]
        vid_name = self.id_to_vidName[vid_id]
        v_name = vid_name[:-4]
        # print(v_name)
        base_path=self.root_path +v_name+'.npy'  # /data/data1/modenet40/modenet40_rename/all/

        out_points=np.load(base_path)#[:512]
        label = self.id_to_action[vid_id]
        
        return out_points, v_name, label


    def __len__(self):
        return len(self.vid_ids)


    def load_data(self):
        self.point_data = []
        for cloud_idx in tqdm(range(self.num_clouds), "Getting video info"):
            self.point_data.append(self.get_pointdata(cloud_idx))

        return self.point_data

    def get_pointdata(self, vid_id):
    
        vid_name = self.point_vids[vid_id]
        # match = re.match(compiled_regex, vid_name)
        # setup, camera, performer, replication, action = [*map(int, match.groups())]
        return {
            'video_cloud_name': vid_name,
            'video_index': vid_id,
            'setup': int(vid_name[-5:-4]),
            'action': int(vid_name[1:4]),
        }
        

    def set_splits(self):
        '''
        Sets the train/test splits
        Cross-Subject Evaluation:
            Train ids = 1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27,
                        28, 31, 34, 35, 38
        Cross-View Evaluation:
            Train camera views: 2, 3
        '''
        # Save the dataset as a dataframe
        dataset = pd.DataFrame(self.point_data)

        # Get the train split ids
        mode_net_train = [0]
        
        # Cross Set splits
        self.train_split_set = list(
            dataset[dataset.setup.isin(mode_net_train)]['video_index'])
        self.test_split_set = list(
            dataset[~dataset.setup.isin(mode_net_train)]['video_index'])