from distutils import text_file
from enum import Flag
from lib2to3.pgen2 import token
import os
import tqdm
import torch
import re
import collections
import imageio
import random
import cv2
import imageio
from tqdm import tqdm

from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import scipy.io as sio
import json
import open_clip
# import spacy

names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

fx = 365.481
fy = 365.481
cx = 257.346
cy = 210.347
# rose@ntu.edu.sg
sample_num_level1 = 64
sample_num_level2 = 64
NUM_POINT = 512
TRAIN_IDS_60 = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
# TRAIN_IDS_60=[1, 2]

TRAIN_IDS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49,
             50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98,
             100, 103]
TRAIN_VALID_IDS = ([1, 2, 5, 8, 9, 13, 14, 15, 16, 18, 19, 27, 28, 31, 34, 38], [4, 17, 25, 35])
compiled_regex = re.compile('.*S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3}).*')
SAMPLE_NUM = 2048
TRAIN_SET = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

class NTU_RGBD_depth(Dataset):
    """NTU depth human masked datasets"""

    def __init__(self, root_path, opt,
                 full_train=True,
                 test=False,
                 validation=False,
                 DATA_CROSS_VIEW=True,
                 Transform=False,
                 DATA_CROSS_SET = False,
                 if_NV = False,
                 if_sequence = False,
                 clip_train = False
                 
                 ):
        
        self.clip_train = clip_train

        self.DATA_CROSS_VIEW = DATA_CROSS_VIEW
        self.root_path = root_path
        self.SAMPLE_NUM = opt.SAMPLE_NUM
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM
        self.transform = Transform
        # self.depth_path = opt.depth_path
        self.exf = opt.exf
        self.if_NV_AUG = if_NV
        self.test = test
        # self.if_test = opt.if_test
        self.if_sequence = if_sequence
        self.tta = opt.tta
        

        self.mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
        self.std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))
        # self.mapping_list = self.get_pair_mapping()

        self.point_vids = os.listdir(self.root_path)
        self.point_vids.sort()

        self.TRAIN_IDS = TRAIN_IDS
        if opt.dataset == 'ntu60':
            indx = self.point_vids.index('S017C003P020R002A060.npy')# S004C001P003R001A001.npy# 'S017C003P017R002A002.npy"#('S017C003P020R002A060.npy')
            # indx = int(0.05*len(self.point_vids))  S017C003P020R002A060
            self.point_vids = self.point_vids[0:indx]
            self.TRAIN_IDS = TRAIN_IDS_60
        
        
        # self.point_vids = self.remove_no_text_samples(self.point_vids)
        
        self.num_clouds = len(self.point_vids)
        print(self.num_clouds)
        
        sup_text = os.listdir('/data/data2/userdata/tanbo/tanbo/ntu_rgb/simply_text_token/token_aug1/')
        self.sup_text_name = sup_text
        self.sup_text_name.sort()
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
        # self.all_text = json.load(open('/data/data2/userdata/tanbo/tanbo/code/depthmap_base/point_base/test_predict.json'))
        self.split_j = self.split_joint()
        self.joint_label_list, self.num_joint_class = self.get_lable_list(max_idex=6)
        self.point_clouds = np.empty(shape=[self.SAMPLE_NUM, self.INPUT_FEATURE_NUM], dtype=np.float32)
        
        self.lookup_table = np.load('../data/lookup_table.npy')

    def __getitem__(self, idx):
        vid_id = self.vid_ids[idx]
        vid_name = self.id_to_vidName[vid_id]
        v_name = vid_name[:20]
        label = self.id_to_action[vid_id]
        
        if self.clip_train:
            depth_im, token_raw = self.read_single_depth_text(v_name)
            return token_raw, 0, depth_im, 0, v_name, label
        
        
        # print(v_name)
        ex_feature = self.exf
        
        modal_choose = list(str(ex_feature))
        # concat text
        key_points_path = '../data/others/' + v_name+'_key.npy'
        points_path = '../data/raw/' + v_name + '.npy'
        
        
        # target_f, local_f =  self.read_feature(v_name) # ******************************************************** for test
        # return target_f, local_f

        if modal_choose[0] =='1':
            points = np.load(points_path)  # [:self.SAMPLE_NUM]
            key_points = np.load(key_points_path)  # [:self.SAMPLE_NUM]
            out_points = self.deal_data_4(points[:, :], key_points[:, :], num_crop=2)
            
            if self.tta:
                out_points2 = self.deal_data_4(points[:, :], key_points[:, :], num_crop=2)
                out_points3 = self.deal_data_4(points[:, :], key_points[:, :], num_crop=2)
                out_points4 = self.deal_data_4(points[:, :], key_points[:, :], num_crop=2)
            
        else:
            out_points = 0
        
        if modal_choose[1] =='1':
            if self.opt.round==1:
                token_raw, token_aug, _, _ = self.concat_text_gettetxt(v_name, if_key=False)
            else:
                token_raw, token_aug = self.expand_text(v_name)
        else:
            token_raw, token_aug =0, 0
        
        if modal_choose[2] =='1':
            # rgb_image = self.get_rgb(v_name)
            rgb_image, appreance_point= self.get_depth_map(v_name)
            
            if self.tta:
                rgb_image2, _ = self.get_depth_map(v_name)
                rgb_image3, _ = self.get_depth_map(v_name)
                rgb_image4, _ = self.get_depth_map(v_name)
                
        else:
            rgb_image, appreance_point= 0, 0
            
        # if modal_choose[3] =='1':
            
        #     appreance_point= appreance_point
        # else:
        #    appreance_point
        
        
        label = self.id_to_action[vid_id]
        if self.tta:
            return  rgb_image2, out_points2, rgb_image, out_points, v_name, label
        
        return  token_raw, token_aug, rgb_image, out_points, v_name, label
    
        
    
        
    def get_rgb(self, v_name):
        rgb_path = '../data/NTU120_depth/ntu_rgb/ntu120_rgb_image/' + v_name + '_rgb.avi/'
        all_imgs = os.listdir(rgb_path)
        all_imgs.sort()
        id_raw = np.arange(0, len(all_imgs)-1, 2)
        id_aug = np.arange(1, len(all_imgs), 2)
        
        # id_raw = np.arange(0, len(all_imgs)//2, 1)
        # id_aug = np.arange(len(all_imgs)//2, len(all_imgs), 1)

        image_raw = []
        for idd in id_raw:
            temp_img = cv2.imread(rgb_path + '/' + str(idd).zfill(3) + '.png')
            temp_img = cv2.resize(temp_img, (128, 128))
            image_raw.append(temp_img)

        image_aug = []
        for idd in id_aug:
            temp_img = cv2.imread(
                rgb_path + '/' + str(idd).zfill(3) + '.png')  # h*w*3
            temp_img = cv2.resize(temp_img, (128, 128))  # 128 * 128 *3
            image_aug.append(temp_img)

        image_raw = np.array(image_raw)  # 5 * 128 * 128 * 3
        image_aug = np.array(image_aug)

        images = np.concatenate((image_raw, image_aug),
                                0)  # (2 * 5)  * 12 * 128 *3

        return images

    
    def read_single_depth_text(self, v_name):
        
        text_path = '../data/text/'+v_name + '_rgb.avi/'
        text_for_video = os.listdir(text_path)
        text_for_video.sort()
        rand_id = np.random.randint(0, len(text_for_video))
        
        with open(text_path + text_for_video[rand_id]) as ff:
            text = ff.read()
        token_raw = open_clip.tokenize(text).squeeze().reshape(-1).numpy()
        
        
        depth_path = '../data/NTU120_depth/nturgb+d_depth_masked/' + v_name
        depth_file = os.listdir(depth_path)
        depth_file.sort()
        depth_id = int(rand_id/len(text_for_video)*len(depth_file))
        
        depth_im = imageio.imread(os.path.join(depth_path, depth_file[depth_id]))  # im is a numpy array

        x, y = np.where(depth_im != 0)
        if x.min()<x.max() and y.min()<y.max():
            depth_im = depth_im[x.min():x.max(), y.min():y.max()]
            depth_im = (depth_im- depth_im.min())/(depth_im.max() - depth_im.min())
            depth_im = depth_im[:, :, np.newaxis]
        
        depth_im = np.tile(depth_im, (1, 1, 3))
        depth_im = (depth_im - self.mean_I) / self.std_I
        depth_im = cv2.resize(depth_im, (224, 224))
        depth_im = depth_im.transpose(2,0,1)
        
        return depth_im, token_raw
    
    
    def get_apprearance(self, points1, points2):
        points1 = points1.reshape(1, -1, 3)
        points2 = points2.reshape(1, -1, 3)
        idex = np.random.randint(0, points1.shape[-2], 2048)
        points1 = points1[:, idex, :]
        
        idex = np.random.randint(0, points2.shape[-2], 2048)
        points2 = points2[:, idex, :]
        
        appreance = np.concatenate((points1, points2), 0)   # 
        return appreance
    


    def get_depth_map(self, v_name):
        point_1 = np.load('../data/NTU120_depth/point_cloud/' + v_name + '.npy') # 16384*2 * 3
        point_2 = np.load("../data/NTU120_depth/ntu_points/" + v_name + '.npy').reshape(-1, 3)  # 24 * 512 * 3

        # point2depth(self, points, factorc = 224):
        deep_data = self.rotate_trans(point_1.reshape(1, -1, 3)).squeeze()
        deep_data_2 = self.rotate_trans(point_2.reshape(1, -1, 3)).squeeze()
        
        appreance_point = self.get_apprearance(deep_data, deep_data_2)
        

        depth_1 = self.point2depth(deep_data, factorc = 224)  # 224 * 224  * 1
        depth_2 = self.point2depth(deep_data_2, factorc = 224)  # 224 * 224  * 1
        
        depth_1 = np.tile(depth_1, (1, 1, 3))
        depth_2 = np.tile(depth_2, (1, 1, 3))
        
        depth_1 = (depth_1 - self.mean_I) / self.std_I
        depth_2 = (depth_2 - self.mean_I) / self.std_I

        depth_1 = depth_1.reshape(1, depth_1.shape[0], depth_1.shape[1], depth_1.shape[2])
        depth_2 = depth_2.reshape(1, depth_2.shape[0], depth_2.shape[1], depth_2.shape[2])

        out_depth_map = np.concatenate((depth_1, depth_2), 0)
        # out_depth_map = np.tile(out_depth_map, (1, 1, 1, 3))

        return out_depth_map, appreance_point

        


    def nomalize_points(self, points_xyzc):
        SAMP = points_xyzc.shape[0]
        ## Normalization
        y_len = points_xyzc[:, 1].max() - points_xyzc[:, 1].min()
        x_len = points_xyzc[:, 0].max() - points_xyzc[:, 0].min()
        z_len = points_xyzc[:, 2].max() - points_xyzc[:, 2].min()

        x_center = (points_xyzc[:, 0].max() + points_xyzc[:, 0].min()) / 2
        y_center = (points_xyzc[:, 1].max() + points_xyzc[:, 1].min()) / 2
        z_center = (points_xyzc[:, 2].max() + points_xyzc[:, 2].min()) / 2
        centers = np.tile([x_center, y_center, z_center], (SAMP, 1))

        anchor = np.tile([points_xyzc[:, 0].min(), points_xyzc[:, 1].min(), points_xyzc[:, 2].min()], (SAMP, 1))

        sca =  y_len if y_len>x_len else x_len

        points_xyzc[:, 0:3] = (points_xyzc[:, 0:3] - anchor) / sca
        return points_xyzc
    
    def point2depth(self, points, factorc = 224):

        # points: batch * numpoints * dim
        # points= np.load("/data/data1/NTU120_depth/point_cloud/S001C001P001R001A002.npy")
        points = self.nomalize_points(points)

        N, K = points.shape
        projction_map = np.zeros((factorc, factorc, 1), dtype=np.float32)
        
        temp = points.copy()
        mx = np.max(temp[:,  2])
        mn = np.min(temp[:,  2])

        temp[:, : 2] = temp[:, : 2] * (factorc-1)
        temp[:, 2] = (temp[:, 2]-mn)/(mx - mn)
        temp[:, 2] = temp[:, 2]*254
        temp = temp.astype(np.int32)

        projction_map[temp[:, 1], temp[:, 0], 0] = temp[ :, 2]
        return projction_map


    
    def get_dynamic_map(self, v_name):
        depth_path = '/data/data1/NTU120_depth/dynamic_image/' + v_name + '/'
        all_depth = os.listdir(depth_path)
        all_depth.sort()
        # print(v_name, len(all_depth))
        all_list = np.arange(0, len(all_depth))
        frame_index = random.sample(list(all_list), 2)
        all_di = []
        for fr_dep in frame_index:
            temp_di = cv2.imread(os.path.join(depth_path, all_depth[fr_dep]))
            temp_di = cv2.resize(temp_di, (224, 224))
            all_di.append(temp_di)
        all_di = np.array(all_di) # 2 * 224 * 224 * 3
        return all_di
        
    
     
    def get_simplify(self, all_key_word):
        text_ = ''
        for sub in all_key_word:
            text_ = text_+','+sub
        text_ = text_.split(',')
        # text_ = np.unique(text_)
        res = []
        [res.append(x) for x in text_ if x not in res]
        text_=''
        for sub in res:
            text_ = text_+','+sub
        return text_[1:]


    def read_lables(self, v_name):
        all_key_words = []
        for sub_im in range(10):
            label_path = "/data/data1/NTU120_depth/labels/"+v_name + '_00' + str(sub_im)+'.txt'
            with open(label_path, 'r') as f:
                temp_text = f.readlines()
            description = ''
            for su_text in temp_text:
                su_text = su_text.split(' ')
                description = description+','+names[int(su_text[0])]
            temp_description = description[1:]   
            all_key_words.append(temp_description)
        return all_key_words
    
    
    def transfer_text2token(self, all_key_words):
        id_raw = np.arange(0, 10-1, 2)
        id_aug = np.arange(1, 10, 2)
        text_raw = ''
        for i in id_raw:
            text_raw = text_raw +',' + all_key_words[i]
        text_raw = text_raw[1:]
        token_raw = open_clip.tokenize(text_raw).squeeze().reshape(-1).numpy()

        text_aug = ''
        for i in id_aug:
            text_aug = text_aug +',' + all_key_words[i]
        text_aug = text_aug[1:]    
        token_aug = open_clip.tokenize(text_aug).squeeze().reshape(-1).numpy()
        return token_raw, token_aug
        
    
    def get_pair_mapping(self, path = '/data/data1/ntu/CVPR2023/feature/point_text_roud1_/pair_table.npy'):   
        pair_table = np.load(path)
        lookup_table = np.load('/data/data1/ntu/CVPR2023/feature/point_text_roud1_/lookup_table.npy')
        
        dic = {}
        count = np.zeros((pair_table.shape[0]), dtype=np.int)
        for i in range(90, lookup_table.shape[0]-300):
            if pair_table[lookup_table[i, 0]] ==0:
                continue
            
            if str(pair_table[lookup_table[i, 0]]) in dic.keys():
                # print('alredy in ', pair_table[lookup_table[i, 0]], lookup_table[i, 0])
                continue
            
            a = np.where(lookup_table[:, 0]==pair_table[lookup_table[i, 0]])
            
            if lookup_table[a[0][0], 1] > lookup_table[i, 1]:
                dic[str(pair_table[lookup_table[i, 0]])] = pair_table[lookup_table[i, 0]]
            else:
                dic[str(pair_table[lookup_table[i, 0]])] = lookup_table[i, 0]
            
        return dic
    
    def mapping_token(self, token):
        maping_list = self.mapping_list
        for i in range(token.shape[0]):
            temp_token = token[i]
            if temp_token in maping_list.keys():
                token[i] = maping_list[temp_token]
        return token
                

    def get_fine_tue_text(self, v_name, if_su_text = False):     
        
        # maping_table = np.load('/data/data1/ntu/CVPR2023/feature/point_text_roud1_/maping_table.npy', allow_pickle='TRUE')
        
        token_raw = np.load('/data/data2/userdata/tanbo/tanbo/ntu_rgb/simply_text_token/token_aug1/'+ v_name + '.npy')
        token_aug = np.load('/data/data2/userdata/tanbo/tanbo/ntu_rgb/simply_text_token/token_aug2/'+ v_name + '.npy')
        if if_su_text:
            end_token_l = np.where(token_raw ==49407)
            
            su_idex = np.load('/data/data2/userdata/tanbo/tanbo/ntu_rgb/simply_text_token/similar_text_index_2/' + v_name + '.npy')  # similar_text_index_round2  similar_text_index_2
            
            if end_token_l[0][0]==1:
                for su_i in range(1, 15):
                    temp_sup_token = np.load('/data/data2/userdata/tanbo/tanbo/ntu_rgb/simply_text_token/token_aug1/' + self.sup_text_name[su_idex[su_i]])
                    end_token_t = np.where(temp_sup_token ==49407)
                    if end_token_t[0][0]==1:
                        continue
                    if end_token_l[0][0]+end_token_t[0][0]>10:
                        continue
                    token_raw[end_token_l[0][0]:end_token_l[0][0]+end_token_t[0][0]] = temp_sup_token[1:end_token_t[0][0]+1]
                    end_token_l = np.where(token_raw ==49407)
            
            end_token_l = np.where(token_aug ==49407)
            su_idex = np.load('/data/data2/userdata/tanbo/tanbo/ntu_rgb/simply_text_token/similar_text_index_round2/' + v_name + '.npy')
            if end_token_l[0][0]==1:
                for su_i in range(1, 10):
                    temp_sup_token = np.load('/data/data2/userdata/tanbo/tanbo/ntu_rgb/simply_text_token/token_aug2/' + self.sup_text_name[su_idex[su_i]])
                    end_token_t = np.where(temp_sup_token ==49407)
                    if end_token_t[0][0]==1:
                        continue
                    if end_token_l[0][0]+end_token_t[0][0]>10:
                        continue
                    token_aug[end_token_l[0][0]:end_token_l[0][0]+end_token_t[0][0]] = temp_sup_token[1:end_token_t[0][0]+1]
                    end_token_l = np.where(token_aug ==49407)
            
            # token_raw = self.mapping_token(token_raw)
            # token_aug = self.mapping_token(token_aug)
                
            
        return token_raw, token_aug

    def remove_no_text_samples(self, all_path):
        # all_path = os.listdir(path)
        all_path.sort()
        clear_list= []
        for v_name in all_path:
            token_raw = np.load('/data/data2/userdata/tanbo/tanbo/ntu_rgb/simply_text_token/token_aug1/'+ v_name)
            end_token_l = np.where(token_raw ==49407)
            if end_token_l[0][0]>1:
                clear_list.append(v_name)
        
        return clear_list
        
        
    
    def sample_frames(self, points, num_frames = 4):
        frame_idx = random.sample(range(0, points.shape[0]), num_frames)
        points = points[frame_idx]
        return points
    
    def split_joint(self):
        splits_j = []
        splits_j.append([20,1,0])
        splits_j.append([8,9,10,11, 23,24])
        splits_j.append([4,5,6,7, 21,22])
        splits_j.append([2,3])
        splits_j.append([12,13,14,15])
        splits_j.append([16,17,18,19])
        return splits_j

    
    def augment_point(self, points, key_point, num_crop): 
        batch_size = 1
        N, C = points.shape
        points = points.reshape(1, N,C)
        key_point = key_point.reshape(1, N,C)

        key_point[:, :, 0:3] = self.jitter_point_cloud(key_point[:, :, 0:3])
        points[:, :, 0:3] = self.jitter_point_cloud(points[:, :, 0:3])

        if np.random.rand()>0.5:
            points = self.reverse_transform(points)

        points = self.rotate_trans(points)
        key_point = self.rotate_trans(key_point)

        points = points.reshape(-1, 128, C)
        key_point = key_point.reshape(-1, 128, C)

        data_pairs = np.concatenate((points, key_point), 0) # (2 * 64) * 128 * 7
        return data_pairs
    
    def get_lable_list(self, max_idex=6):
        lable_list = {}
        labell_id = 0
        for i in range(max_idex):
            for j in range(i+1, max_idex):
                key = str(i).zfill(2) + str(j).zfill(2)# + str(k).zfill(2)
                lable_list[key] = labell_id
                labell_id+=1
        return lable_list, len(lable_list)


    def rand_drop_joint(self, points_input, drop_num = 2, samples_joints = 64):
        ''' input : frames * dim * joint * 128'''
        label_list, num_class = self.joint_label_list, self.num_joint_class
        split_joints = self.split_j
        points = points_input.copy()
        B, D, J,  K = points.shape
        label_for_joint_class = []
        # for b_idex in range(B):
        
        rand_drop_idx = random.sample(range(0, len(split_joints)), drop_num)
        rand_drop_idx.sort()

        temp_key_id = str(rand_drop_idx[0]).zfill(2) + str(rand_drop_idx[1]).zfill(2)
        rand_drop_idx = np.array(rand_drop_idx).reshape(-1)
        temp_label = label_list[temp_key_id]
        # label_for_joint_class.append(temp_label)

        for slice_id in range(drop_num):
            points[:, :, split_joints[rand_drop_idx[slice_id]], :] = 0
        
        # keep joint points
        joint_points = points[:, :, :25, :].copy()    
        no_joint_points = points[:, :, 25: , :].copy()   
        
        # delete drop_points
        new_list = []
        for i in rand_drop_idx:
            new_list.append(split_joints[i])
        new_li = [c for a in new_list for c in a]
        drops = np.array(new_li)
        joint_points = np.delete(joint_points, drops ,2)
        
        joint_points = joint_points.transpose(0,2,3,1).reshape(-1, K, D)
        no_joint_points = no_joint_points.transpose(0, 2, 3, 1).reshape(-1, K, D)

        if joint_points.shape[0]>samples_joints:
            joint_points = joint_points[:samples_joints]
        else:
            joint_points = np.concatenate((joint_points, no_joint_points[:samples_joints-joint_points.shape[0]]), 0)   # 64 * 128 * 7
        
        return joint_points, temp_label
    
    
    def get_motion(self, points):
        F, D, J, K = points.shape
        motion_value  =np.zeros((F, 1, J, K), dtype=float)
        for i in range(F):
            motion_value[i] = (i-0.5*F)/F
        motion_points = np.concatenate((points, motion_value), 1)
        return motion_points    
        

    def build_qa(self, text):
        pharase = text.split(' , ')
        num_phrase = len(pharase)
        if num_phrase<=2:
            return text, text
        mask_idx = np.random.randint(2,num_phrase)
        answer = pharase[mask_idx]
        question = ''
        for ii in range(1,num_phrase):
            if ii!=mask_idx:
                question = question+' , '+pharase[ii]
        return question, answer

    def test_SpaCy(self, text):
    
        nlp = spacy.load('en_core_web_sm')
        
        doc = nlp(text)
        
        token = [token.text for token in doc]
        return doc, token
    
    def build_question_N_V(self, text_input, str_text, n_r, v_r):
        fin_tex  = ''
        n = ''
        v = ''
        temp = text_input.split(' ')
        add_temp = []
        doc, token = self.test_SpaCy(str_text)
        for token in doc:
            if token.text in temp:
                continue
            if token.pos_=="NOUN" or token.pos_=="VERB":
                fin_tex = fin_tex+ ' '+token.text
            if token.pos_=="NOUN":
                n = n+ ' '+token.text
            if token.pos_=="VERB":
                v = v+ ' '+token.text
        if fin_tex =='':
            return text_input, n_r, v_r 
        return text_input+' , ' +  fin_tex, n_r +  ' , '+ n, v_r +  ' , '+ v
    
    
    
    def new_caption(self, v_name):
        base_path = '/data/data1/NTU120_depth/ntu_text/totall_caption/'+v_name + '_rgb.avi/'
        all_files = os.listdir(base_path)
        all_files.sort()
        
        id_raw = np.arange(0, len(all_files)-1, 2)
        id_aug = np.arange(1, len(all_files), 2)
        
        
        with open(base_path + all_files[id_raw[0]]) as ff:
            strat_stage = ff.read()
        
        frame_change = ''
        for text_id in id_raw:
            with open(base_path + all_files[text_id]) as ff:
                temp = ff.read()
            
            t_change = self.get_dynamic(temp, strat_stage)
            if t_change =='':
                continue
            frame_change = frame_change +', '+ t_change
            strat_stage = temp
        token_raw = frame_change
        
        
        with open(base_path + all_files[id_aug[0]]) as ff:
            strat_stage = ff.read()
        
        frame_change = ''
        for text_id in id_aug:
            with open(base_path + all_files[text_id]) as ff:
                temp = ff.read()
            
            t_change = self.get_dynamic(temp, strat_stage)
            if t_change =='':
                continue
            frame_change = frame_change +', '+ t_change
            strat_stage = temp
        token_aug = frame_change
        
        return token_raw, token_aug
           
            
            
    def get_dynamic(self, caption, phrase):
        words = caption.split(' ')
        p_after = phrase.split(' ')
        change = ''
        for word in words:
            if word in p_after:
                continue
            change = change+' '+ word
        return change
    

    def build_question(self, caption, phrase):
        words = caption.split(' ')
        p_after = phrase.split(' ')
        question = ''
        for word in words:
            if word in p_after:
                continue
            question = question+' '+ word
        if question =='':
            return phrase
        return phrase+' , '+ question
    
    
    def read_all_text(self, v_name):
        text_path = '/data/data1/NTU120_depth/ntu_text/totall_caption/'+v_name + '_rgb.avi/'
        text_for_video = os.listdir(text_path)
        text_for_video.sort()
        all_caption_token = []
        
        for i in range(len(text_for_video)):
            with open(text_path + text_for_video[i]) as ff:
                temp = ff.read()
                temp = open_clip.tokenize(temp).squeeze().reshape(-1).numpy()
                all_caption_token.append(temp)
        
        all_caption_token = np.array(all_caption_token)
        if all_caption_token.shape[0]<10:
            idd = np.random.randint(0, all_caption_token.shape[0], 10-all_caption_token.shape[0])
            all_caption_token = np.concatenate((all_caption_token, all_caption_token[idd]), 0)
        return all_caption_token
    

        
        # return depth_im
        
        

    
    
    def concat_text_gettetxt(self, v_name, if_key = False, old = False):
        
        text_path = '../data/text/'+v_name + '_rgb.avi/'
        text_for_video = os.listdir(text_path)
        text_for_video.sort()
        text_raw = ''
        text_aug = ''
        
        id_raw = np.arange(0, len(text_for_video)-1, 2)
        id_aug = np.arange(1, len(text_for_video), 2)
        for i in id_raw:
    
            with open(text_path + text_for_video[i]) as ff:
                temp = ff.read()
                
            text_raw = text_raw +' , ' + temp
        text_raw = text_raw[3:]

        token_raw = open_clip.tokenize(text_raw).squeeze().reshape(-1).numpy()

        for i in id_aug:
            with open(text_path + text_for_video[i]) as ff:
                temp = ff.read() 
            text_aug = text_aug +' , ' + temp 
        text_aug = text_aug[3:]
             
        token_aug = open_clip.tokenize(text_aug).squeeze().reshape(-1).numpy()
        if if_key:
            token_raw =  self.fittertxt(token_raw)
            token_aug =  self.fittertxt(token_aug)

        return token_raw, token_aug, text_raw, text_aug
    
    
    def fittertxt(self, token_raw, start =5, end = 6): 
        id = np.random.randint(start, end, size = 1)
        new_token = []
        out_token = np.zeros_like(token_raw)[:77]
        out_token[0]  = token_raw[0]
        for t_i in token_raw: 
            if t_i ==49406:
                continue
            if t_i == 49407:
                continue   
            if t_i in self.lookup_table[id[0]:-300, 0]:
                # continue
                new_token.append(t_i)
        new_token = np.array(new_token).reshape(-1)
        new_token = np.unique(new_token)
        out_token[1:new_token.shape[0]+1] = new_token
        out_token[new_token.shape[0]+1] = 49407
        
        return out_token
    
    def expand_text(self, v_name):
        token_all_raw = []
        token_all_aug = []
        
        token_raw, token_aug, _, _ = self.concat_text_gettetxt(v_name, if_key = False)
        token_all_raw.append(token_raw)
        token_all_aug.append(token_aug)
        
        idexx = np.where(token_raw == 49407)
    # if idexx[0] ==2:
        ###****************记得把采样点数修改回去！！！！！！
        id_nums = random.sample(list(range(1, 30)), 20)
        # for i in id_nums:
        for i in range(1, 1+3):
            # temp_name = np.load('/data/data2/userdata/tanbo/tanbo/ntu_rgb/for_paper/ntu60/cv/index1/' + v_name + '_' + str(i) + '.npy')
            
            temp_name = np.load('../data/neibor_indexs/' + str(self.opt.round-1).zfill(3) + v_name + '/' + str(i).zfill(3) + '.npy')
            temp_name = str(temp_name)[:-4]
            token_raw, token_aug, _, _ = self.concat_text_gettetxt(temp_name, if_key = False)
            token_all_raw.append(token_raw)
            token_all_aug.append(token_aug)
        
        token_all_raw = np.array(token_all_raw).reshape(-1)
        token_all_aug = np.array(token_all_aug).reshape(-1)
        
        token_all_raw = self.fittertxt(token_all_raw, start= 0, end=1)
        token_all_aug = self.fittertxt(token_all_aug, start= 0, end=1)
        
        # token_all_raw = self.fittertxt(token_all_raw, start= 40, end=60)
        # token_all_aug = self.fittertxt(token_all_aug, start= 40, end=60)
        
        return token_all_raw, token_all_aug
    
    
    def concat_text(self, v_name, if_key = False, old = False):      
        _, _, text_raw, text_aug = self.concat_text_gettetxt(v_name, if_key = False, old = True)
        _, _, text_raw_new, text_aug_new = self.concat_text_gettetxt(v_name, if_key = False, old = False)
        text_raw = text_raw + ' , ' + text_raw_new
        text_aug = text_aug + ' , ' + text_aug_new
        token_raw = open_clip.tokenize(text_raw).squeeze().reshape(-1).numpy()
        token_aug = open_clip.tokenize(text_aug).squeeze().reshape(-1).numpy()
        return token_raw, token_aug
        
    
    
    
    def deal_text(self, v_name):
        if self.test:
            text_path = '/data/data1/NTU120_depth/test_caption/' + v_name + '/'
        else:
            text_path = '/data/data1/NTU120_depth/caption/' + v_name + '/'
        text_for_video = os.listdir(text_path)
        text_for_video.sort()
        text_raw = ''
        text_aug = ''
        
        n_r  = ''
        v_r = ''
        text_raw_nv = ''
        
        n_a  = ''
        v_a = ''
        text_aug_nv = ''
        
        id_raw = np.arange(0, len(text_for_video)-1, 2)
        id_aug = np.arange(1, len(text_for_video), 2)
        for i in id_raw:
            with open(text_path + text_for_video[i]) as ff:
                temp = ff.read()
            text_raw = self.build_question(temp, text_raw)

        # # #*********************************NV aug*************************************    
        # #     if self.if_NV_AUG:
        # #         text_raw_nv, n_r, v_r= self.build_question_N_V(text_raw_nv, temp, n_r, v_r)
        # if self.if_NV_AUG:
        #     with open('/data/data1/NTU120_depth/caption_new/' + v_name + '/raw_n.txt') as ff:
        #         text_raw = ff.read()
        # # #*****************************************************************************    

        token_raw = open_clip.tokenize(text_raw).squeeze().reshape(-1).numpy()

        for i in id_aug:
            with open(text_path + text_for_video[i]) as ff:
                temp = ff.read() 
            text_aug = self.build_question(temp, text_aug)
        
        # # #*********************************NV aug*************************************    
        # #     if self.if_NV_AUG:
        # #         text_aug_nv, n_a, v_a= self.build_question_N_V(text_aug_nv, temp, n_a, v_a)
        # if self.if_NV_AUG:
        #     with open('/data/data1/NTU120_depth/caption_new/' + v_name + '/aug_n.txt') as ff:
        #         text_aug = ff.read()
        #  #***********************************************************************    
         
             
        token_aug = open_clip.tokenize(text_aug).squeeze().reshape(-1).numpy()

        return token_raw, token_aug#, Q_raw, A_raw, Q_aug, A_aug

    def deal_data_4(self, points, key_point, num_crop): 
        batch_size = 1
        N, C = points.shape
        points = points.reshape(1, N,C)
        key_point = key_point.reshape(1, N,C)
        
        
        if self.if_sequence:
            points = self.get_temporal_augment(points)
            key_point = self.get_temporal_augment(key_point)
        

        points_new, key_point_new = self.points_sample_jiter(points, key_point)
        
        if self.if_sequence:
            points[:, :NUM_POINT, :] = self.fps_sample_data_single(points_new, sample_num_level1)
            key_point[:, :NUM_POINT, :] = self.fps_sample_data_single(key_point_new, sample_num_level1)
        else:
            points[:, :NUM_POINT, :] = self.fps_sample_data(points_new, sample_num_level1, sample_num_level2)
            key_point[:, :NUM_POINT, :] = self.fps_sample_data(key_point_new, sample_num_level1, sample_num_level2)
        if np.random.rand()>0.5:
            points = self.reverse_transform(points)
        
        

        deep_data = self.rotate_trans(points[:, :, :4])
        deep_data_2 = self.rotate_trans(key_point[:, :, :4])

        data_pairs = np.empty([num_crop, 2048, 4], dtype=float)
        data_pairs = np.concatenate((deep_data, deep_data_2), 0)    # (crop*T) * 2048 * 4

        return data_pairs


    def get_text(self, video_name):
        video_all = os.listdir(self.root_path + video_name)
        video_all.sort()
        num_frames = len(video_all)
        # all_sam = np.arange(num_frames)
        index = [0, int(num_frames/4),int(num_frames/2), int(num_frames*3/4),num_frames-1]
        index = np.array(index)
        raw_points = []

        for frame in index:
            depth_im = imageio.imread(self.root_path + video_name + '/' + video_all[frame])
            # raw_img.append(depth_im)
            raw_point_cloud = self.depth_to_pointcloud(depth_im.copy()).reshape(-1, 3)
            raw_points.append(raw_point_cloud)

        # all_points = np.array(raw_points)     
        index = np.array([0,2,4])
        base_points = self.get_points_from_list(raw_points.copy()).reshape(1, -1, 3)
        time_points = self.get_points_from_list(raw_points[0:3].copy()).reshape(1, -1, 3)

        base_points = self.rotate_trans(base_points).reshape(-1, 3)
        time_points = self.rotate_trans(time_points).reshape(-1, 3)
        time_points[:, 0] = -time_points[:, 0]

        base_points = self.normalize_point(base_points.reshape(-1, 3))
        time_points = self.normalize_point(time_points.reshape(-1, 3))

        raw_img = self.points_to_img(base_points)
        aug_img = self.points_to_img(time_points)

        base_points = self.remove_same_data(base_points, 512*1).reshape(-1, 3)
        time_points = self.remove_same_data(time_points, 512*1).reshape(-1, 3)

        return raw_img, aug_img, base_points, time_points        
        
    
    def video_to_depth_map(self, video_name):
        video_all = os.listdir(self.root_path + video_name)
        video_all.sort()
        num_frames = len(video_all)
        # all_sam = np.arange(num_frames)
        index = [0, int(num_frames/4),int(num_frames/2), int(num_frames*3/4),num_frames-1]
        index = np.array(index)
        raw_points = []

        for frame in index:
            depth_im = imageio.imread(self.root_path + video_name + '/' + video_all[frame])
            # raw_img.append(depth_im)
            raw_point_cloud = self.depth_to_pointcloud(depth_im.copy()).reshape(-1, 3)
            raw_points.append(raw_point_cloud)

        # all_points = np.array(raw_points)     
        index = np.array([0,2,4])
        base_points = self.get_points_from_list(raw_points.copy()).reshape(1, -1, 3)
        time_points = self.get_points_from_list(raw_points[0:3].copy()).reshape(1, -1, 3)

        base_points = self.rotate_trans(base_points).reshape(-1, 3)
        time_points = self.rotate_trans(time_points).reshape(-1, 3)
        time_points[:, 0] = -time_points[:, 0]

        base_points = self.normalize_point(base_points.reshape(-1, 3))
        time_points = self.normalize_point(time_points.reshape(-1, 3))

        raw_img = self.points_to_img(base_points)
        aug_img = self.points_to_img(time_points)

        base_points = self.remove_same_data(base_points, 512*1).reshape(-1, 3)
        time_points = self.remove_same_data(time_points, 512*1).reshape(-1, 3)

        return raw_img, aug_img, base_points, time_points

    def get_points_from_list(self, points_list):
        temp = points_list[0]
        for frames in range(1, len(points_list)):
            temp = np.concatenate((temp, points_list[frames]), 0)
        return temp


    def points_to_img(self, points, img_size=224):
        #input points: numpoints * 3
        depth_img = np.zeros((img_size, img_size, 3), dtype=np.float32)
        points[:, 0] = -points[:, 0]
        points[:, 1] = points[:, 1]
        
        points[:, 0]= points[:, 0]- points[:, 0].min()
        points[:, 1]= points[:, 1]- points[:, 1].min()
        points[:, 2]= points[:, 2]- points[:, 2].min()
        # points[:, 0]= (points[:, 0]*223).astype(np.int)
        # points[:, 1]= (points[:, 1]*223).astype(np.int)
        points = points*(img_size-1)
        points = points.astype(np.int)
        depth_img[points[:, 1], points[:, 0], 0] = points[:, 2]
        depth_img[points[:, 1], points[:, 0], 1] = points[:, 2]
        depth_img[points[:, 1], points[:, 0], 2] = points[:, 2]
        
        return depth_img

    def normalize_point(self, points):
        x_len = points[:, 0].max() - points[:, 0].min()
        y_len = points[:, 1].max() - points[:, 1].min()
        z_len = points[:, 2].max() - points[:, 2].min()
        len_points = np.max([x_len, y_len])

        x_center = (points[:, 0].max() + points[:, 0].min())/2
        y_center = (points[:, 1].max() + points[:, 1].min())/2
        z_center = (points[:, 2].max() + points[:, 2].min())/2
        center = np.tile([x_center, y_center, z_center], (points.shape[0], 1))
        points = (points-center)/len_points
        return points


    def depth_to_pointcloud(self, depth_im):
        rows,cols = depth_im.shape
        xx,yy = np.meshgrid(range(0,cols), range(0,rows))

        valid = depth_im > 0
        xx = xx[valid]
        yy = yy[valid]
        depth_im = depth_im[valid]

        X = (xx - cx) * depth_im / fx
        Y = (yy - cy) * depth_im / fy
        Z = depth_im


        # x = X/Z*fx +cx
        # y = Y/Z*fy +cy

        points3d = np.array([X.flatten(), Y.flatten(), Z.flatten()])
        points3d = points3d.transpose(1,0)
        return points3d

    def pointcloud_to_depth(self, points3d, depth_im):
        row, col = points3d.shape
        projection_img = np.zeros_like(depth_im)
        X, Y, Z = points3d[:, 0], points3d[:, 1], points3d[:, 2]
        xx = X/Z*fx +cx
        yy = Y/Z*fy +cy
        zz = Z

        xx = np.ceil(xx*row/xx.max())
        yy = np.ceil(yy*col/yy.max())


        projection_img[xx[:], yy[:]] = zz
        return projection_img


    def get_random_augment(self, points):
        reverse = np.random.randint(0, 2)
        temp_points = points.copy()
        temp_points = self.rotate_trans(temp_points)
        if reverse == 1:
            temp_points = self.reverse_transform(temp_points)
        
        temp_points[:, :, :3] = self.jitter_point_cloud(temp_points[:, :, :3])

        return temp_points

    def remove_same_data(self, points, scale = 100, npoints = 2048):
        num_points, dim = points.shape
        # after_sample_points = np.zeros((batch, NUM_POINT, dim), dtype=np.float)
        # for batch_i in range(batch):
        temp = points.copy()
        #remove same points
        temp = temp*scale
        temp = temp.astype(np.int)
        temp = np.unique(temp, axis=0)
        temp = temp/scale
        # sample points to NUM_POINT
        idex_2 = np.random.randint(0, temp.shape[0], size=npoints)
        temp = temp[idex_2]
        return temp
    
    
    def get_temporal_augment(self, pointss):
        points = pointss.copy().squeeze()
        # print(points.shape)
        out_points = []
        for temporal_int in range(3, points.shape[-1]):
            point_temporal = np.concatenate((points[:, 0:3].copy(), points[:, temporal_int:temporal_int+1].copy()), axis = 1)
            idexx = np.where(point_temporal[:, 3]!=0)
            point_temporal = point_temporal[idexx]
            if point_temporal.shape[0]==0:
                point_temporal = points[:, 0:4].copy()
            else:
                idex = np.random.randint(0, point_temporal.shape[0], 2048)
                point_temporal = point_temporal[idex]
            out_points.append(point_temporal)
        out_points = np.array(out_points)
        return out_points                       # T, 2048 * 4
        
    
    
    def get_temporal_augment_data(self, pointss, temporal_int):
        points = pointss.copy()
        point_temporal = np.concatenate((points[:, 0:3].copy(), points[:, temporal_int:temporal_int+1].copy()), axis = 1)
        idexx = np.where(point_temporal[:, 3]!=0)
        point_temporal = point_temporal[idexx]
        idex = np.random.randint(0, point_temporal.shape[0], 512)
        point_temporal = point_temporal[idex]
        # point_temporal = self.jitter_point_cloud(point_temporal)
        # point_temporal = self.get_fps_data(point_temporal)
        return point_temporal
    
    def fps_sample_data(self, points_xyzc, sample_num_level1, sample_num_level2):
        # NUM_POINT = 512
        for kk in range(points_xyzc.shape[0]):
            sampled_idx_l1 = self.farthest_point_sampling_fast(points_xyzc[kk, :, 0:3], sample_num_level1)
            other_idx = np.setdiff1d(np.arange(NUM_POINT), sampled_idx_l1.ravel())
            new_idx = np.concatenate((sampled_idx_l1.ravel(), other_idx))
            points_xyzc[kk, :, :] = points_xyzc[kk, new_idx[:NUM_POINT], :]
        return points_xyzc
    
    
    def fps_sample_data_single(self, points_xyzc, sample_num_level1):
        # NUM_POINT = 512
        sampled_idx_l1 = self.farthest_point_sampling_fast(points_xyzc[0, :, 0:3], sample_num_level1)
        other_idx = np.setdiff1d(np.arange(NUM_POINT), sampled_idx_l1.ravel())
        new_idx = np.concatenate((sampled_idx_l1.ravel(), other_idx))
        
        for kk in range(points_xyzc.shape[0]):
            points_xyzc[kk, :, :] = points_xyzc[kk, new_idx[:NUM_POINT], :]
        return points_xyzc


    def farthest_point_sampling_fast(self, pc, sample_num):
        pc_num = pc.shape[0]

        sample_idx = np.zeros(shape=[sample_num, 1], dtype=np.int32)
        sample_idx[0] = np.random.randint(0, pc_num)

        cur_sample = np.tile(pc[sample_idx[0], :], (pc_num, 1))
        diff = pc - cur_sample
        min_dist = (diff * diff).sum(axis=1)

        for cur_sample_idx in range(1, sample_num):
            ## find the farthest point

            sample_idx[cur_sample_idx] = np.argmax(min_dist)
            if cur_sample_idx < sample_num - 1:
                diff = pc - np.tile(pc[sample_idx[cur_sample_idx], :], (pc_num, 1))
                min_dist = np.concatenate((min_dist.reshape(pc_num, 1), (diff * diff).sum(axis=1).reshape(pc_num, 1)), axis=1).min(axis=1)  ##?
        # print(min_dist)
        return sample_idx


    def points_sample_jiter(self, points, key_point):
        idex = np.random.randint(0, points.shape[1], size=NUM_POINT)
        points = points[:, idex, :]
        idex = np.random.randint(0, key_point.shape[1], size=NUM_POINT)
        key_point = key_point[:, idex, :]
        key_point[:, :, 0:3] = self.jitter_point_cloud(key_point[:, :, 0:3])
        points[:, :, 0:3] = self.jitter_point_cloud(points[:, :, 0:3])
        return points, key_point


    def reverse_transform(self, points):
        reverse_data = np.zeros(points.shape, dtype=np.float32)
        reverse_data[:, :, :] = points[:, :, :]
        reverse_data[:, :, 0] = -reverse_data[:, :, 0]
        # reverse_data[:, :, 0:3] = self.jitter_point_cloud(reverse_data[:, :, 0:3])
        return reverse_data


    def depth_transform(self, points,  angle_set):
        rotated_data = np.zeros(points.shape, dtype=np.float32)
        rotated_data[:, :, :] = points[:, :, :]
        for k in range(rotated_data.shape[0]):
            # depth transform
            # rotate

            angle =  angle_set * np.pi* 0.25
            Ry = np.array([[np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]])
            shape_pc = rotated_data[k, :, 0:3]
            # print(shape_pc.shape)
            rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), Ry)
        # normalize
        # nor_data = normalize_data(rotated_data)
        return rotated_data
    
    def rotate_point_cloud_x(self, batch_data, roatate_set=1):
        """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
        BxNx3 array, original batch of point clouds
        Return:
        BxNx3 array, rotated batch of point clouds
        """
        rotated_data = batch_data.copy()
        for k in range(batch_data.shape[0]):
            rotation_angle =  (np.random.rand()+0.4) * roatate_set * np.pi/6
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[1, 0, 0],
                                        [0, cosval, -sinval],
                                        [0, sinval, cosval]])
            # shape_pc = batch_data[k, ...]
            # rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
            shape_pc = rotated_data[k, :, 0:3]
            # print(shape_pc.shape)
            rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
            
            # shape_pc = rotated_data[k, :, 3:6]
            # rotated_data[k, :, 3:6] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        return rotated_data


    def rotate_trans(self, points):
        # rotated_data = np.zeros(points.shape, dtype=np.float32)
        rotated_data = points.copy()
        rotated_data = self.rotate_point_cloud_x(rotated_data)
        for k in range(rotated_data.shape[0]):

            angle =  (np.random.rand()-0.5) * np.pi
            Ry = np.array([[np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]])
            shape_pc = rotated_data[k, :, 0:3]
            # print(shape_pc.shape)
            rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), Ry)
            
            # shape_pc = rotated_data[k, :, 3:6]
            # rotated_data[k, :, 3:6] = np.dot(shape_pc.reshape((-1, 3)), Ry)
        # normalize
        # nor_data = normalize_data(rotated_data)
        return rotated_data
    
    def rank_transform(self, points, rank_slop=-1):
        rank_data = np.zeros(points.shape, dtype=np.float32)
        rank_data[:, :, :] = points[:, :, :]
        rank_data = rank_slop * rank_data
        # rank_data=rank_data-rank_data.min()
        return rank_data
    
    def scale_trans(self, points):
        
        rank_slop = np.random.rand()+0.5
        rank_data = points.copy()
        rank_data[:, :, :3] = rank_slop *  rank_data[:, :, :3]
        # rank_data=rank_data-rank_data.min()
        return rank_data

    def jitter_point_cloud(self, batch_data, sigma=0.01, clip=0.05):
        """ Randomly jitter points. jittering is per point.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, jittered batch of point clouds
        """
        B, N, C = batch_data.shape
        assert (clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
        jittered_data += batch_data
        return jittered_data
    
    

    def __len__(self):
        return len(self.vid_ids)


    def load_data(self):
        self.point_data = []
        for cloud_idx in tqdm(range(self.num_clouds), "Getting video info"):
            self.point_data.append(self.get_pointdata(cloud_idx))

        return self.point_data

    def get_pointdata(self, vid_id):
    
        vid_name = self.point_vids[vid_id]
        # print(vid_name)
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

