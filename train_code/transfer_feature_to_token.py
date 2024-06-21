import numpy as np
from collections import Counter
import os
import open_clip
import torch.nn.functional as F
import torch
import pickle
import json
train_ids_camera = [2, 3]
def read_text():
    v_name = 'S001C001P003R001A002'
    text_path = '/data/data1/NTU120_depth/ntu_text/totall_caption/'+v_name + '_rgb.avi/'
    text_for_video = os.listdir(text_path)
    text_for_video.sort()
    with open(text_path + text_for_video[6]) as ff:
        temp = ff.read() 
    token_raw = np.load('../ntu_rgb/simply_text_token/token_aug1/'+ v_name + '.npy')
    print(temp, token_raw[:20])

# def show_similarity():
#     base_path =  '../ntu_rgb/simply_text_token/similar_text_index/'
#     all_file = os.listdir(base_path)
#     all_file.sort()
#     for sub_fi in all_file:
#         cros_file = np.load('../ntu_rgb/simply_text_token/similar_text_index/'+sub_fi)
#         print(sub_fi, 'crosbuding to: ', all_file[cros_file[0]], all_file[cros_file[1]], all_file[cros_file[2]])
#         print('//')

def Mergers_tokens():
    all_token = '../ntu_rgb/simply_text_token/token_aug_round2_1/'
    all_files = os.listdir(all_token)
    all_files.sort()
    all_files = all_files#[:1000]
    pair_table = np.zeros((49408, 49408), dtype=np.int)
    
    ccc = 0
    for sub_f in all_files:
        ccc +=1
        temp = np.zeros((77), dtype=np.long)
        temp_i = 0
        token = np.load(os.path.join(all_token, sub_f))
        for token_i in range(1, token.shape[0]):
            if token[token_i] == 49407:
                break
            if token[token_i] in temp[:temp_i]:
                continue
            temp[temp_i] = token[token_i]
            temp_i +=1
        temp = temp[:temp_i]
        
        for i in range(0, temp.shape[0]-1):
            for j in range(i+1 , temp.shape[0]):
                pair_table[temp[i], temp[j]]+=1
        
        print(sub_f, ccc)
    # print(pair_table[1937].sum(),pair_table[1937, 1519], pair_table[1519].sum(), pair_table[8225].sum())
    pair_idex = np.argmax(pair_table, 1)
    print(pair_idex.shape)
    np.save('../feature/point_text_roud1_/pair_table.npy', pair_idex)

def get_pair_mapping(path = '../feature/point_text_roud1_/pair_table.npy'):   
    pair_table = np.load(path)
    lookup_table = np.load('../feature/point_text_roud1_/lookup_table.npy')
    
    dic = {}
    count = np.zeros((pair_table.shape[0]), dtype=np.int)
    for i in range(100, lookup_table.shape[0]-300):
        if pair_table[lookup_table[i, 0]] ==0:
            continue
        
        if str(pair_table[lookup_table[i, 0]]) in dic.keys():
            print('alredy in ', pair_table[lookup_table[i, 0]], lookup_table[i, 0])
            continue
        
        a = np.where(lookup_table[:, 0]==pair_table[lookup_table[i, 0]])
        
        if lookup_table[a[0][0], 1] > lookup_table[i, 1]:
            dic[str(pair_table[lookup_table[i, 0]])] = pair_table[lookup_table[i, 0]]
        else:
            dic[str(pair_table[lookup_table[i, 0]])] = lookup_table[i, 0]
            
    
    with open('../feature/point_text_roud1_/maping_table.json', "w") as tf:
        json.dump(dic,tf)
        
    with open('../feature/point_text_roud1_/maping_table.pkl', "r") as tf:
        a = json.load(tf)
    # np.save('../feature/point_text_roud1_/maping_table.npy', dic)
    # a = np.load('../feature/point_text_roud1_/maping_table.npy', allow_pickle='TRUE')
    print('')
    # a = np.where(count>1)
    # print(a.shape)
            

        
def conput_num():
    idex = 0
    base_path =  '../ntu_rgb/simply_text_token/token_aug2/'
    all_file = os.listdir(base_path)
    all_file.sort()
    al_num= len(all_file)
    for sub_fi in all_file:
        cros_file = np.load(base_path+sub_fi)
        en_token = np.where(cros_file==49407)
        if en_token[0][0] ==1:
            idex =idex+1
    print(idex, idex/al_num)


# def finding_more_text_via_point(save_path = "../data/neibor_indexs/", point_feature_path = " last round learned point cloud feature  save path", round = 0):

#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     all_files = os.listdir(point_feature_path)
#     all_files.sort()
#     DIMM = 512*2
#     print('totlly samples:', len(all_files))
#     all_files = all_files#[:10000]
#     que_feature = []
#     file_name = []
#     for sub_file in all_files:
#         view = int(sub_file[5:8])
#         if view in train_ids_camera:
#             temp_feature= np.load(point_feature_path + sub_file)[:DIMM]
#             que_feature.append(temp_feature)
#             file_name.append(sub_file)
#     que_feature = np.array(que_feature)
#     que_feature = torch.from_numpy(que_feature)
#     que_feature = F.normalize(que_feature, p=2, dim=1).numpy()
    
#     all_files = os.listdir(point_feature_path)
#     all_files.sort()
#     print('totlly samples:', len(all_files))
#     idd = 0
#     for sub_file in all_files:
#         view = int(sub_file[5:8])
#         if view in train_ids_camera:
#             continue
#         print(sub_file, idd)
#         idd +=1
#         temp_feature= np.load(point_feature_path + sub_file)[:DIMM].reshape(1, -1)
#         temp_feature = torch.from_numpy(temp_feature)
#         temp_feature = F.normalize(temp_feature, p=2, dim=1).numpy()
#         similarity = temp_feature@(que_feature.transpose(1, 0)) # 1 * K
#         # indexx = np.argmax(similarity)
#         indexx = np.argsort(-similarity)
#         np.save(save_path+sub_file, file_name[indexx[0, 1]])
#         # np.save(save_path+sub_file, indexx[0, 0:20])    # similar_text_index      similar_text_index_r. similar_text_index_r
#         print(sub_file, 'crosbuding to: ', all_files[indexx[0, 1]], all_files[indexx[0, 2]], all_files[indexx[0, 3]])
        

def token_text(v_name):
    text_path = '../data/text/'+v_name
    text_for_video = os.listdir(text_path)
    text_for_video.sort()
    text_raw = ''
    for i in np.arange(len(text_for_video)):
        with open(text_path + text_for_video[i]) as ff:
            temp = ff.read()
            
        text_raw = text_raw +' , ' + temp
    text_raw = text_raw[3:]
    token_raw = open_clip.tokenize(text_raw).squeeze().reshape(-1).numpy()
    return token_raw


def fiter_high_freequency():  # findding the top 100 words
    token_path = '../data/text/'
    all_files = os.listdir(token_path)
    all_files.sort()
    print('totlly samples:', len(all_files))
    all_files = all_files

    token = []
    for sub_file in all_files:
        token_feature= token_text(sub_file)
        token.append(token_feature)
    token = np.array(token)
    print(token.shape)
    token = token.reshape(-1)
    print('all token is :', token.shape)
    frequen = Counter(token).most_common(10000)
    frequen = np.array(frequen)
    np.save('../data/lookup_table.npy', frequen)
    print(frequen.shape)
    

def demo_of_text():
    text_raw = 'a woman standing in a room holding a wii cup drinking'
    token_raw = open_clip.tokenize(text_raw).squeeze().reshape(-1).numpy()
    # tok = open_clip.deco(text_raw)
    
    text_ra = 'a woman standing in a room holding a cup'
    token_ra = open_clip.tokenize(text_ra).squeeze().reshape(-1).numpy()
    print('finish')

def finding_strong_response(id_sample, num_samp = 20000): # remove the start token
    look_up_table = np.load('../data/lookup_table.npy')
    feature_path = '../data/saved_feature/'
    all_files = os.listdir(feature_path)
    all_files.sort()
    # all_files = all_files[id_sample*num_samp:(id_sample+1)*num_samp]
    i = 0
    for sub_file in all_files:
        print(sub_file, i)
        i = i+1
        watch_idex(sub_file, look_up_table)



def watch_idex(v_name, look_up_table, key_words_num = 5):
    new_token_for_save = np.zeros((77, ), dtype = np.long)
    new_token_for_save[0] = 49406
    
    local_feature = np.load('/data/data2/ntu/CVPR2023/feature/point_text_roud2_/feature/' + v_name)[:, 1:, :] #frmas * 77 * 512
    token_feature= np.load('/data/data2/ntu/CVPR2023/feature/point_text_roud2_/token/' + v_name)[:, 1:]
    save_path = '../ntu_rgb/simply_text_token/token_aug_round2_1/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    max_frequence = np.zeros((local_feature.shape[0], key_words_num))
    load_idex = np.argsort(-local_feature, 1)  # frames * 77 * 512
    for frames_id in range(0, local_feature.shape[0]):
        frequen = Counter(load_idex[frames_id, 0, :]).most_common(key_words_num)
        for key_id in range(0, key_words_num):
            max_frequence[frames_id, key_id] = token_feature[frames_id, frequen[key_id][0]]
        # print(frames_id ,max_frequence[frames_id, :])
        
    max_frequence = max_frequence.reshape(-1).astype(np.int)
    pt_r = 1
    for ii in range(0, max_frequence.shape[0]):
        if pt_r<new_token_for_save.shape[0]-1:
            if max_frequence[ii] in look_up_table[:90, 0]:   # 89   50
                continue
            
            if max_frequence[ii] in look_up_table[-500:, 0]:  #-400   -500
                continue
    
            # if max_frequence[ii] in new_token_for_save:
            #     continue
            new_token_for_save[pt_r] = max_frequence[ii]
            pt_r = pt_r+1
    new_token_for_save[pt_r] = 49407
    np.save(save_path+v_name, new_token_for_save)

def cal_idex(local_idex):
    calulate = np.zeros((77), dtype = np.int)
    for iii in range(local_idex.shape[-1]):
        calulate[local_idex[0,0, iii]] +=1
    print(calulate)
    # print(np.around(100*calulate[:]/calulate[:].sum(),1))

# watch_idex()
# demo_of_text()
# show_similarity()

fiter_high_freequency()       # find the high frequence token for text simplification
# finding_strong_response(id_sample=2)  # get the similarity 
# finding_more_text_via_point(round = 0)         # text extention 
# Mergers_tokens()
# get_pair_mapping()
# read_text()


# conput_num()
