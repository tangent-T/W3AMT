import os
import tqdm
import imageio
import numpy as np
import time
import random
import torch
import logging

'''
due to the ntu120 full depth maps data is not avialable, the action proposal is used by skeleton-based action proposal.
'''

fx = 365.481
fy = 365.481
cx = 257.346
cy = 210.347

voxel_size = 30
M = 5  # temporal splits

Key_NUM = 16
Keep_NUM = 8
Negative_NUM = 88

two_person_action_idex = np.arange(50, 61)
SAMPLE_NUM = 2048
sample_num_level1 = 512
sample_num_level2 = 128
K = 60  # max frame limit for temporal rank
boundary_seg = [60, 29 , 10 ,10] # lx,rx,ty,by
key_save_path = '../data/others/'  # '/3DV_pointdata/NTU_voxelsize35_split5'
save_path = '../data/raw/'
app_save_path = '../data/app/'
if not os.path.exists(key_save_path): 
    os.makedirs(key_save_path)
    os.makedirs(save_path)
    os.makedirs(app_save_path)

try:
    os.makedirs(save_path)
except OSError:
    pass

def append_points(all_frame_points_list, vo_DI, min_x, min_y, min_z):
    # input: num_points * 3
    num_f = len(all_frame_points_list)
    if num_f<10:
        frame_choose = np.random.randint(0, num_f, 10).tolist()
    else:
        frame_choose = np.arange(0, num_f, 1).tolist()
    frame_choose.sort()
    # frame_choose = [int(0.1*num_f), int(0.4*num_f),int(0.7*num_f),int(0.95*num_f)]
    all_append_points = []
    for f_i_app in frame_choose:
        # print(f_i_app)
        itm = all_frame_points_list[f_i_app].T
        all = np.zeros((itm.shape[0], 4), dtype= np.float)
        index0 = ((itm[:, 0] - min_x) / voxel_size).astype(np.int32)
        index1 = ((itm[:, 1] - min_y) / voxel_size).astype(np.int32)
        index2 = ((itm[:, 2] - min_z) / voxel_size).astype(np.int32)

        motion = vo_DI[0, index0, index1, index2] # num_points * 1
        
        all[:, 0] = (itm[:, 0] - min_x) / voxel_size
        all[:, 1] = (itm[:, 1] - min_y) / voxel_size
        all[:, 2] = (itm[:, 2] - min_z) / voxel_size
        all[:, 3] = motion
        all_append_points.append(all)
    return all_append_points


def main():
    data_path = '../ntu120dataset'
    sub_Files = os.listdir(data_path)
    sub_Files.sort()


    for s_fileName in sub_Files:
        # s_fileName='nturgbd_depth_masked_s002'
        videoPath = os.path.join(data_path, s_fileName, 'nturgb+d_depth_masked')
        if os.path.isdir(videoPath):
            print(s_fileName)
            video_Files = os.listdir(videoPath)
            # print(video_Files)
            video_Files.sort()

            for video_FileName in video_Files:
                print(video_FileName)
                SAMPLE_NUM = 2048
                idex_of_action = int(video_FileName[-3:])

                filename = video_FileName + '.npy'
                filename_key_for_all = video_FileName + '_key.npy'
                filename_app = video_FileName + '_app.npy'

                file = os.path.join(save_path, filename)
                # if os.path.isfile(file):
                #     continue

                pngPath = os.path.join(videoPath, video_FileName)
                imgNames = os.listdir(pngPath)
                imgNames.sort()
                # print(imgNames)

                ## ------ select a fixed number K of images
                n_frame = len(imgNames)
                all_sam = np.arange(n_frame)

                if n_frame > K:
                    frame_index = random.sample(list(all_sam), K)
                    # frame_index = np.array(frame_index)
                    n_frame = K
                else:
                    frame_index = all_sam.tolist()

                frame_index.sort()

                ### ------convert the depth sequence to points data
                each_frame_points_num = np.zeros(n_frame, dtype=np.int32)
                each_frame_points_num_for_location = np.zeros(n_frame, dtype=np.int32)
                all_frame_points_list = []
                all_frame_points_list_for_location = []

                temp_path = imgNames[0]
                img_path = os.path.join(pngPath, temp_path)
                depth_im_pr = load_depth_from_img(img_path)
                depth_im_pr = depth_im_pr.astype(np.int32)
                i = 0
                for i_frame in frame_index:
                    depthName = imgNames[i_frame]
                    img_path = os.path.join(pngPath, depthName)
                    depth_im_cu = load_depth_from_img(img_path)
                    temp_differ = np.zeros(depth_im_cu.shape, np.int32)
                    temp_differ, depth_im_pr = locate_motion(depth_im_pr, depth_im_cu)
                    cloud_im_for_location = depth_to_pointcloud(temp_differ, idex_of_action)
                    all_frame_points_list_for_location.append(cloud_im_for_location)  # all frame points in 1 list
                    each_frame_points_num_for_location[i] = cloud_im_for_location.shape[1]

                    depth_im = load_depth_from_img(img_path)
                    cloud_im = depth_to_pointcloud(depth_im, idex_of_action)
                    # print(depth_im.shape,cloud_im.shape)
                    all_frame_points_list.append(cloud_im)  # all frame points in 1 list
                    each_frame_points_num[i] = cloud_im.shape[1]
                    # print(cloud_im.shape[1])
                    i = i + 1
                if each_frame_points_num.sum() == 0:
                    logging.info('{} --{} '.format('in Valid', video_FileName))
                    continue
                all_frame_points_array = np.zeros(shape=(3, each_frame_points_num.sum()))
                # print(each_frame_points_num.sum())

                ## compute the Max bounding box to voxelization
                for i_frame in range(n_frame):
                    start_idx = each_frame_points_num[0:i_frame].sum()
                    all_frame_points_array[:, start_idx:start_idx + each_frame_points_num[i_frame]] = \
                    all_frame_points_list[i_frame]
                # print(all_frame_points_array.shape)
                max_x = all_frame_points_array[0, :].max()
                max_y = all_frame_points_array[1, :].max()
                max_z = all_frame_points_array[2, :].max()
                min_x = all_frame_points_array[0, :].min()
                min_y = all_frame_points_array[1, :].min()
                min_z = all_frame_points_array[2, :].min()

                dx, dy, dz = map(int, [(max_x - min_x) / voxel_size, (max_y - min_y) / voxel_size,(max_z - min_z) / voxel_size])

                voxel_DI, voxel_DI_key = get_modify_rankpooling_point(dx, dy, dz, n_frame, min_x, min_y, min_z,
                                                                      all_frame_points_list,
                                                                      all_frame_points_list_for_location, M=5)

                # voxel_DI_key[0, :, :, :] = discad_volxe(voxel_DI_key[0, :, :, :])
                # voxel_DI_key[0, :, :, :] = discad_volxe(voxel_DI_key[0, :, :, :], 1)
                # voxel_DI_key[0, :, :, :] = disca_voxel(voxel_DI_key[0, :, :, :],9)
                voxel_DI_key[0, :, :, :] = disca_voxel(voxel_DI_key[0, :, :, :], 6)
                voxel_DI[0, :, :, :] = disca_voxel(voxel_DI[0, :, :, :], 5)

                appen_points = append_points(all_frame_points_list, voxel_DI, min_x, min_y, min_z) # 4 * (points * 4)

                ### 3DV voxel to 3DV points
                mm, xx, yy, zz = np.where(voxel_DI[:, :, :, :] != 0)
                xyz = np.column_stack((xx, yy, zz))
                if len(xx) > SAMPLE_NUM:
                    xyz = np.unique(xyz, axis=0)
                motion = voxel_DI[:, xyz[:, 0], xyz[:, 1], xyz[:, 2]]
                points_xyzc = np.concatenate((xyz, motion.T),axis=1)  # final 3DV point feature shape N*(x,y,z,m_g,m_1,...m_t)

                ### Sample and normalization
                if len(xx) < SAMPLE_NUM:
                    rand_points_index_for_all = np.random.randint(0, points_xyzc.shape[0], size=SAMPLE_NUM - len(xx))
                    points_xyzc = np.concatenate((points_xyzc, points_xyzc[rand_points_index_for_all, :]), axis=0)
                else:
                    rand_points_index_for_all = np.random.randint(0, points_xyzc.shape[0], size=SAMPLE_NUM)
                    points_xyzc = points_xyzc[rand_points_index_for_all, :]

                ### for all key point
                mm, xx, yy, zz = np.where(voxel_DI_key[:, :, :, :] != 0)
                new_voxel_DI_key = np.zeros(shape=[M, dx + 1, dy + 1, dz + 1])
                new_voxel_DI_key[:, xx, yy, zz] = voxel_DI[:, xx, yy, zz]
                mm, xx, yy, zz = np.where(new_voxel_DI_key[:, :, :, :] != 0)
                xyz_key = np.column_stack((xx, yy, zz))
                if len(xx) > SAMPLE_NUM:
                    xyz_key = np.unique(xyz_key, axis=0)
                motion_key = new_voxel_DI_key[:, xyz_key[:, 0], xyz_key[:, 1], xyz_key[:, 2]]
                points_xyzc_key_all = np.concatenate((xyz_key, motion_key.T),
                                                     axis=1)  # final 3DV point feature shape N*(x,y,z,m_g,m_1,...m_t)

                ### Sample and normalization for key
                if len(xx) < SAMPLE_NUM:
                    rand_points_index = np.random.randint(0, points_xyzc_key_all.shape[0], size=SAMPLE_NUM - len(xx))
                    points_xyzc_key_all = np.concatenate((points_xyzc_key_all, points_xyzc_key_all[rand_points_index, :]), axis=0)
                else:
                    rand_points_index = np.random.randint(0, points_xyzc_key_all.shape[0], size=SAMPLE_NUM)
                    points_xyzc_key_all = points_xyzc_key_all[rand_points_index, :]

                ## Normalization
                y_len = points_xyzc[:, 1].max() - points_xyzc[:, 1].min()
                x_len = points_xyzc[:, 0].max() - points_xyzc[:, 0].min()
                z_len = points_xyzc[:, 2].max() - points_xyzc[:, 2].min()
                c_max, c_min = points_xyzc[:, 3:8].max(axis=0), points_xyzc[:, 3:8].min(axis=0)
                c_len = c_max - c_min

                x_center = (points_xyzc[:, 0].max() + points_xyzc[:, 0].min()) / 2
                y_center = (points_xyzc[:, 1].max() + points_xyzc[:, 1].min()) / 2
                z_center = (points_xyzc[:, 2].max() + points_xyzc[:, 2].min()) / 2
                centers = np.tile([x_center, y_center, z_center], (SAMPLE_NUM, 1))

                points_xyzc[:, 0:3] = (points_xyzc[:, 0:3] - centers) / y_len
                points_xyzc[:, 3:8] = (points_xyzc[:, 3:8] - c_min) / c_len - 0.5

                points_xyzc_key_all[:, 0:3] = (points_xyzc_key_all[:, 0:3] - centers) / y_len
                points_xyzc_key_all[:, 3:8] = (points_xyzc_key_all[:, 3:8] - c_min) / c_len - 0.5

                frame_appear = np.zeros((len(appen_points), SAMPLE_NUM, 4), dtype=np.float)
                for f_i_ap in range(len(appen_points)):
                    frame_points = appen_points[f_i_ap].copy()
                    if frame_points.shape[0] < SAMPLE_NUM:
                        rand_index = np.random.randint(0, frame_points.shape[0], size=SAMPLE_NUM - frame_points.shape[0])
                        frame_appear[f_i_ap] = np.concatenate((frame_points, frame_points[rand_index, :]), axis=0)
                    else:
                        rand_index = np.random.randint(0, frame_points.shape[0], size=SAMPLE_NUM)
                        frame_appear[f_i_ap] = frame_points[rand_index, :]

                    frame_appear[f_i_ap, :, 0:3] = (frame_appear[f_i_ap, :, 0:3] - centers) / y_len
                    frame_appear[f_i_ap, :, 3:] = (frame_appear[f_i_ap, :, 3:]-c_min[0])/c_len[0] - 0.5         

                save_npy(points_xyzc, filename, 1)
                save_npy(points_xyzc_key_all, filename_key_for_all, 2)
                save_npy(frame_appear, filename_app, 3)


def save_npy(data, filename, mode=1):
    if mode == 2:
        file = os.path.join(key_save_path, filename)
    elif mode == 1:
        file = os.path.join(save_path, filename)
    elif mode == 3:
        file = os.path.join(app_save_path, filename)

    np.save(file, data)

def disca_voxel(voxel_DI, th_discard = 8):
    row, col, dep = voxel_DI.shape
    t_emp = np.zeros((row, col, dep), dtype=np.int)
    tx, ty,tz = np.where(voxel_DI[:, :,:] != 0)
    t_emp[tx, ty, tz] = 1
    emp = np.zeros((row-2, col-2, dep-2), dtype=np.int)

    for x_i in range(-1, 2):
        for y_i in range(-1, 2):
            for z_i in range(-1, 2):
                emp = emp+ t_emp[1+x_i:row-1+x_i, 1+y_i:col-1+y_i, 1+z_i:dep-1+z_i]

    raw_x, raw_y, raw_z = np.where(voxel_DI[:, :, :] == 0)

    t_emp[1: row-1, 1:col-1, 1:dep-1] = emp
    xx, yy, zz = np.where(t_emp[:, :, :] < th_discard)

    voxel_DI[xx, yy, zz] = 0
    voxel_DI[raw_x, raw_y, raw_z ] = 0
    return voxel_DI


def farthest_point_sampling_fast(pc, sample_num):
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
            min_dist = np.concatenate((min_dist.reshape(pc_num, 1), (diff * diff).sum(axis=1).reshape(pc_num, 1)),
                                      axis=1).min(axis=1)  ##?
    # print(min_dist)
    return sample_idx


def depth_to_pointcloud(depth_im, idex_of_action):
    rows, cols = depth_im.shape
    xx, yy = np.meshgrid(range(0, cols), range(0, rows))
    valid = depth_im > 0
    xx = xx[valid]
    yy = yy[valid]
    depth_im = depth_im[valid]

    X = (xx - cx) * depth_im / fx
    Y = (yy - cy) * depth_im / fy
    Z = depth_im

    points3d = np.array([X.flatten(), Y.flatten(), Z.flatten()])

    return points3d


# cut the ground ,x>0.7*shape[0] is cutted
def load_depth_from_img(depth_path):
    depth_im = imageio.imread(depth_path)  # im is a numpy array
    depth_im[0:2, :] = 0
    depth_im[-1:-10, :] = 0
    depth_im[:, 0:2] = 0
    depth_im[:, -1:-10] = 0

    x, y = np.where(depth_im != 0)
    depth_im[0: boundary_seg[0], :] = 0
    depth_im[x[-1] - boundary_seg[1]:, :] = 0
    depth_im[:, 0:y.min() + boundary_seg[2]] = 0
    depth_im[:, y.max() - boundary_seg[3]:] = 0
    return depth_im



def locate_motion(pr_image, cu_image):
    up_th = 300
    low_th = 50
    cu_image = cu_image.astype(np.int32)
    xx, yy = np.where((abs(cu_image - pr_image) > low_th) & (abs(cu_image - pr_image) < up_th))
    temp_differ = np.zeros(cu_image.shape, np.int32)
    motion_intense = np.zeros(cu_image.shape, np.int32)
    motion_intense[xx, yy] = abs(cu_image - pr_image)[xx, yy]

    temp_differ[xx, yy] = cu_image[xx, yy]
    pr_image[:, :] = cu_image[:, :]
    return temp_differ, pr_image  # , motion_intense


def get_modify_rankpooling_point(dx, dy, dz, n_frame, min_x, min_y, min_z, all_frame_points_list,
                                 all_frame_points_list_for_location, M=5):
    voxel_DI = np.zeros(shape=[M, dx + 1, dy + 1, dz + 1])
    voxel_DI_for_key = np.zeros(shape=[1, dx + 1, dy + 1, dz + 1])
    # avg_threeD_matrix = np.zeros(shape=[dx+1,dy+1,dz+1], dtype=np.float32)
    for i_frame in range(n_frame):

        # threeD_matrix = np.zeros(shape=[dx + 1, dy + 1, dz + 1], dtype=np.float32)
        # # voxelization
        # itm = all_frame_points_list[i_frame].T
        # index0 = ((itm[:, 0] - min_x) / voxel_size).astype(np.int32)
        # index1 = ((itm[:, 1] - min_y) / voxel_size).astype(np.int32)
        # index2 = ((itm[:, 2] - min_z) / voxel_size).astype(np.int32)
        # threeD_matrix[index0, index1, index2] = 1


        # threeD_matrix = np.zeros(shape=[dx + 1, dy + 1, dz + 1], dtype=np.float32)
        # threeD_matrix_for_key = np.zeros(shape=[dx + 1, dy + 1, dz + 1], dtype=np.float32)
        # ## voxelization
        # for itm in all_frame_points_list[i_frame].T:
        #     threeD_matrix[int((itm[0] - min_x) / voxel_size)][int((itm[1] - min_y) / voxel_size)][int((itm[2] - min_z) / voxel_size)] = 1
        # for itm in all_frame_points_list_for_location[i_frame].T:
        #     threeD_matrix_for_key[int((itm[0] - min_x) / voxel_size)][int((itm[1] - min_y) / voxel_size)][int((itm[2] - min_z) / voxel_size)] = 1

        threeD_matrix = np.zeros(shape=[dx + 1, dy + 1, dz + 1], dtype=np.float32)
        # voxelization
        sacle = 1
        itm = all_frame_points_list[i_frame].T
        index0 = ((itm[:, 0] - min_x) / voxel_size).astype(np.int32)
        index1 = ((itm[:, 1] - min_y) / voxel_size).astype(np.int32)
        index2 = ((itm[:, 2] - min_z) / voxel_size).astype(np.int32)
        threeD_matrix[index0, index1, index2] = 1

        threeD_matrix_for_key = np.zeros(shape=[sacle * (dx + 1), sacle * (dy + 1), sacle * (dz + 1)], dtype=np.float32)
        itm_key = all_frame_points_list_for_location[i_frame].T
        index0 = ((itm_key[:, 0] - min_x)*sacle / voxel_size).astype(np.int32)
        index1 = ((itm_key[:, 1] - min_y)*sacle / voxel_size).astype(np.int32)
        index2 = ((itm_key[:, 2] - min_z)*sacle / voxel_size).astype(np.int32)
        threeD_matrix_for_key[index0, index1, index2] = 1

        voxel_DI_for_key[0, :, :, :] = voxel_DI_for_key[0, :, :, :] + (i_frame * 2 - n_frame + 1) * threeD_matrix_for_key

        for m in range(M):
            ## first segment 3dv construction: all frame voxel is used!   motion:(m_g)
            if m == 0:
                voxel_DI[0, :, :, :] = voxel_DI[0, :, :, :] + (i_frame * 2 - n_frame + 1) * threeD_matrix

            ## T_1=M-1 segments 3dv construction: M-1 temporal splits with 0.5 overlap
            if m == 1 and i_frame < round(n_frame * 2 / 5):
                idx_f = i_frame
                len_f = round(n_frame * 2 / 5)
                voxel_DI[m, :, :, :] = voxel_DI[m, :, :, :] + (idx_f * 2 - len_f + 1) * threeD_matrix
                # voxel_DI_rank[m, :, :, :] = voxel_DI_rank[m, :, :, :] + len_f * (2 /(1+np.exp(len_f-2*idx_f)) - 1) * threeD_matrix

            if m == 2 and i_frame < round(n_frame * 3 / 5) and i_frame >= round(n_frame * 1 / 5):
                idx_f = i_frame - round(n_frame * 1 / 5)
                len_f = round(n_frame * 3 / 5) - round(n_frame * 1 / 5)
                voxel_DI[m, :, :, :] = voxel_DI[m, :, :, :] + (idx_f * 2 - len_f + 1) * threeD_matrix
                # voxel_DI_rank[m, :, :, :] = voxel_DI_rank[m, :, :, :] + len_f * (2 / (1 + np.exp(len_f - 2 * idx_f)) - 1) * threeD_matrix

            if m == 3 and i_frame < round(n_frame * 4 / 5) and i_frame >= round(n_frame * 2 / 5):
                idx_f = i_frame - round(n_frame * 2 / 5)
                len_f = round(n_frame * 4 / 5) - round(n_frame * 2 / 5)
                voxel_DI[m, :, :, :] = voxel_DI[m, :, :, :] + (idx_f * 2 - len_f + 1) * threeD_matrix
                # voxel_DI_rank[m, :, :, :] = voxel_DI_rank[m, :, :, :] + len_f * ( 2 / (1 + np.exp(len_f - 2 * idx_f)) - 1) * threeD_matrix

            if m == 4 and i_frame >= round(n_frame * 3 / 5):
                idx_f = i_frame - round(n_frame * 3 / 5)
                len_f = n_frame - round(n_frame * 3 / 5)
                voxel_DI[m, :, :, :] = voxel_DI[m, :, :, :] + (idx_f * 2 - len_f + 1) * threeD_matrix
                # voxel_DI_rank[m, :, :, :] = voxel_DI_rank[m, :, :, :] + len_f * (2 / (1 + np.exp(len_f - 2 * idx_f)) - 1) * threeD_matrix
    return voxel_DI, voxel_DI_for_key


def discad_volxe(voxel_DI, mode=0):
    row, col, dep = voxel_DI.shape
    xx, yy, zz = np.where(voxel_DI[:, :, :] != 0)
    len_xx = len(xx)
    for i in range(0, len_xx):
        if_isolate = 0
        if xx[i] > 1 and xx[i] < row - 1 and yy[i] > 1 and yy[i] < col - 1 and zz[i] > 1 and zz[i] < dep - 1:
            if voxel_DI[xx[i] + 1, yy[i], zz[i]] == 0:
                if_isolate = if_isolate + 1.5
            if voxel_DI[xx[i] + 1, yy[i] + 1, zz[i]] == 0:
                if_isolate = if_isolate + 1.2

            if voxel_DI[xx[i], yy[i] + 1, zz[i]] == 0:
                if_isolate = if_isolate + 1.5
            if voxel_DI[xx[i] + 1, yy[i] - 1, zz[i]] == 0:
                if_isolate = if_isolate + 1.2

            if voxel_DI[xx[i] - 1, yy[i], zz[i]] == 0:
                if_isolate = if_isolate + 1.5
            if voxel_DI[xx[i] - 1, yy[i] - 1, zz[i]] == 0:
                if_isolate = if_isolate + 1.2

            if voxel_DI[xx[i], yy[i] - 1, zz[i]] == 0:
                if_isolate = if_isolate + 1.5
            if voxel_DI[xx[i] - 1, yy[i] + 1, zz[i]] == 0:
                if_isolate = if_isolate + 1.2

            if voxel_DI[xx[i], yy[i], zz[i] + 1] == 0:
                if_isolate = if_isolate + 1.2
            if voxel_DI[xx[i], yy[i], zz[i] - 1] == 0:
                if_isolate = if_isolate + 1.2

            if voxel_DI[xx[i] - 1, yy[i], zz[i] + 1] == 0:
                if_isolate = if_isolate + 1
            if voxel_DI[xx[i] + 1, yy[i], zz[i] - 1] == 0:
                if_isolate = if_isolate + 1

            if voxel_DI[xx[i] + 1, yy[i], zz[i] + 1] == 0:
                if_isolate = if_isolate + 1
            if voxel_DI[xx[i] - 1, yy[i], zz[i] - 1] == 0:
                if_isolate = if_isolate + 1

            if voxel_DI[xx[i], yy[i] - 1, zz[i] + 1] == 0:
                if_isolate = if_isolate + 1
            if voxel_DI[xx[i], yy[i] + 1, zz[i] - 1] == 0:
                if_isolate = if_isolate + 1

            if voxel_DI[xx[i], yy[i] + 1, zz[i] + 1] == 0:
                if_isolate = if_isolate + 1
            if voxel_DI[xx[i], yy[i] - 1, zz[i] - 1] == 0:
                if_isolate = if_isolate + 1
        th = 13
        if mode == 1:
            th = 14
        elif mode == 2:
            th = 16
        if if_isolate >= th:
            voxel_DI[xx[i], yy[i], zz[i]] = 0
        else:
            continue
    return voxel_DI



if __name__ == '__main__':
    main()
