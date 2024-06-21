import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import numpy as np
import torch.nn.functional as F


def info_nce_loss_pytorch(features, n_views= 2, Tem = 0.07):
    batch_size = int(features.shape[0]/n_views)
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    logits = logits / Tem
    return logits, labels


def group_points_3DV_2048(points, knn_K, sample_num_level1, SAMPLE_NUM=2048):
    # group points using knn and ball query
    # points: B * SAMPLE_NUM * 4
    cur_train_size = points.shape[0]
    INPUT_FEATURE_NUM = points.shape[-1]
    # knn_K = 64
    ball_radius = 0.16

    points = points.view(cur_train_size, -1, INPUT_FEATURE_NUM)
    #print(points.shape)
    inputs1_diff = points[:,:,0:3].transpose(1,2).unsqueeze(1).expand(cur_train_size,sample_num_level1,3,SAMPLE_NUM) \
                 - points[:,0:sample_num_level1,0:3].unsqueeze(-1).expand(cur_train_size,sample_num_level1,3,SAMPLE_NUM)# B * 512 * 3 * 1024
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 128 * 3 * 512
    inputs1_diff = inputs1_diff.sum(2)                      # B * 128 * 512 distance
    dists, inputs1_idx = torch.topk(inputs1_diff, knn_K, 2, largest=False, sorted=False)  # dists: B * 128 * 32; inputs1_idx: B * 128 * 32
        
    # ball query
    invalid_map = dists.gt(ball_radius) # B * 128 * 32  value: binary

    for jj in range(sample_num_level1):
        inputs1_idx[:,jj,:][invalid_map[:,jj,:]] = jj
        
    idx_group_l1_long = inputs1_idx.view(cur_train_size,sample_num_level1*knn_K,1).expand(cur_train_size,sample_num_level1*knn_K,INPUT_FEATURE_NUM)
    #print(points.shape,'points')
    inputs_level1 = points.gather(1,idx_group_l1_long).view(cur_train_size,sample_num_level1,knn_K,INPUT_FEATURE_NUM) # B*128*32*4

    inputs_level1_center = points[:,0:sample_num_level1,0:3].unsqueeze(2)       # B*128*1*3
    inputs_level1[:,:,:,0:3] = inputs_level1[:,:,:,0:3] - inputs_level1_center.expand(cur_train_size,sample_num_level1,knn_K,inputs_level1_center.shape[-1])
    inputs_level1 = inputs_level1.unsqueeze(1).transpose(1,4).squeeze(4)  # B*4*128*32
    inputs_level1_center = inputs_level1_center.contiguous().view(-1,1,sample_num_level1,3).transpose(1,3)  # B*3*128*1

    ##
    # inputs_level1 = inputs_level1.view(-1,opt.temperal_num,opt.INPUT_FEATURE_NUM,opt.sample_num_level1,opt.knn_K)
    # inputs_level1_center = inputs_level1_center.view(-1,opt.temperal_num,3,opt.sample_num_level1,1)

    return inputs_level1, inputs_level1_center



def augment_different(predict, criterion):
    labels = np.arange(0,10,1)
    labels = torch.from_numpy(labels).type(torch.long).cuda()
    loss = criterion(predict, labels)
    return loss
    

def global_contrast(num_crop, x_global, x, opt, criterion):
    
    fix_weight = np.ones((opt.batchSize, opt.batchSize)) - np.identity(opt.batchSize)
    fix_weight_loss = np.tile(fix_weight, (1, num_crop))

    fix_weight_loss = torch.from_numpy(fix_weight_loss).type(torch.FloatTensor).cuda()
    # fix_weight_loss = fix_weight_loss.type(torch.FloatTensor).cuda()

    global_if = 1
    if global_if ==1:
    
        l_pos = torch.Tensor(num_crop, opt.batchSize, 1).cuda()
        loss_all = torch.Tensor(num_crop, 1).cuda()

        for lo_i in range(0,num_crop):
            l_pos[lo_i] = torch.einsum('nc,nc->n', [x_global, x[(lo_i) * opt.batchSize:(lo_i+1) * opt.batchSize, :]]).unsqueeze(-1)

        k1 = x.clone().permute(1, 0)
        l_neg2 = torch.einsum('nc,ck->nk', [x_global, k1])  # batch * (5*batch)
        l_neg2 = torch.mul(l_neg2, fix_weight_loss)

        l_neg2 = l_neg2.repeat(num_crop,1,1)
        logits_p= torch.cat([l_pos, l_neg2], dim=2)

        # labels: positive key indicators
        labels = torch.zeros(logits_p.shape[1], dtype=torch.long).cuda()
        for loss_i in range(num_crop):
            loss_all[loss_i] = criterion(logits_p[loss_i], labels)

        loss_c = (loss_all.sum())
        return loss_c

def circle_contrast(num_crop, x, batchSize, criterion):
    fix_weight = np.ones((batchSize, batchSize)) - np.identity(batchSize)
    fix_weight_circle = np.tile(fix_weight, (1, (num_crop) * (num_crop - 1)))
    fix_weight_circle = torch.from_numpy(fix_weight_circle).type(torch.FloatTensor).cuda()

    
    circle_if = 1
    if circle_if == 1:
        l_pos_circle_all = torch.Tensor(num_crop-1, batchSize, 1).cuda()
        l_neg_circle_all = torch.Tensor(num_crop-1, batchSize, num_crop*batchSize).cuda()

        order_idex = np.arange(0, num_crop, 1)  # 0~9
        np.random.shuffle(order_idex)
        loss_circle_all = torch.Tensor(num_crop-1).cuda()
        for l_pos1_circle_i in range(num_crop-1):
            l_pos_circle_all[l_pos1_circle_i] = torch.einsum('nc,nc->n',[x[order_idex[l_pos1_circle_i] * batchSize:(order_idex[l_pos1_circle_i] + 1) * batchSize, :],x[order_idex[l_pos1_circle_i+1] * batchSize:(order_idex[l_pos1_circle_i+1] + 1) * batchSize,:]]).unsqueeze(-1)

        for l_neg_circle_i in range(num_crop-1):
            l_neg_circle_all[l_neg_circle_i] = torch.einsum('nc,ck->nk', [x[order_idex[l_neg_circle_i] * batchSize:(order_idex[l_neg_circle_i] + 1) * batchSize, :],x.clone().permute(1, 0)])

        l_neg_circle = l_neg_circle_all.permute(1,0,2).reshape(batchSize, -1)
        l_neg_circle = torch.mul(l_neg_circle, fix_weight_circle)

        l_neg_circle = l_neg_circle.repeat(num_crop-1, 1 ,1)
        logits_p_circle = torch.cat([l_pos_circle_all, l_neg_circle], dim=2)

        # labels: positive key indicators
        labels = torch.zeros(logits_p_circle.shape[1], dtype=torch.long).cuda()
        for loss_circle_i in range(num_crop-1):
            loss_circle_all[loss_circle_i] = criterion(logits_p_circle[loss_circle_i], labels)
        loss_circle = loss_circle_all.sum()
        return loss_circle
    
def circle_contrast_neg(num_crop, x, batchSize, criterion):
    fix_weight = np.ones((batchSize, batchSize)) - np.identity(batchSize)
    fix_weight_circle = np.tile(fix_weight, (1, (num_crop) * (num_crop - 1)))
    fix_weight_circle = torch.from_numpy(fix_weight_circle).type(torch.FloatTensor).cuda()

    
    circle_if = 1
    if circle_if == 1:
        l_pos_circle_all = torch.Tensor(num_crop-1, batchSize, 1).cuda()
        l_neg_circle_all = torch.Tensor(num_crop-1, batchSize, num_crop*batchSize).cuda()

        order_idex = np.arange(0, num_crop, 1)  # 0~9
        np.random.shuffle(order_idex)
        loss_circle_all = torch.Tensor(num_crop-1).cuda()
        for l_pos1_circle_i in range(num_crop-1):
            l_pos_circle_all[l_pos1_circle_i] = torch.einsum('nc,nc->n',[x[order_idex[l_pos1_circle_i] * batchSize:(order_idex[l_pos1_circle_i] + 1) * batchSize, :],x[order_idex[l_pos1_circle_i+1] * batchSize:(order_idex[l_pos1_circle_i+1] + 1) * batchSize,:]]).unsqueeze(-1)

        for l_neg_circle_i in range(num_crop-1):
            l_neg_circle_all[l_neg_circle_i] = torch.einsum('nc,ck->nk', [x[order_idex[l_neg_circle_i] * batchSize:(order_idex[l_neg_circle_i] + 1) * batchSize, :],x.clone().permute(1, 0)])

        l_neg_circle = l_neg_circle_all.permute(1,0,2).reshape(batchSize, -1)
        l_neg_circle = torch.mul(l_neg_circle, fix_weight_circle)

        l_neg_circle = l_neg_circle.repeat(num_crop-1, 1 ,1)
        logits_p_circle = torch.cat([l_pos_circle_all, l_neg_circle], dim=2)

        # labels: positive key indicators
        labels = torch.zeros(logits_p_circle.shape[1], dtype=torch.long).cuda()
        for loss_circle_i in range(num_crop-1):
            loss_circle_all[loss_circle_i] = criterion(-logits_p_circle[loss_circle_i], labels)
        loss_circle = loss_circle_all.sum()
        return loss_circle

    
def CLD_Loss(epoch, num_crop, x_nor, opt):
    cld_if = 1
    loss_CLD = 0
    CLD_start = 0
    if cld_if ==1 and epoch>=CLD_start:
        loss_CLD_all = torch.Tensor(num_crop-4).cuda()
        for CLD_i in range(num_crop-4):
            loss_CLD_all[CLD_i] = grouping(x_nor[CLD_i * opt.batchSize: (CLD_i+3) * opt.batchSize], x_nor[(CLD_i+1) * opt.batchSize: (CLD_i+4) * opt.batchSize], 0.05, 10, 60, 5) #k_eigen, clusters, num_iters
        loss_CLD = loss_CLD_all.sum()
    return loss_CLD


def grouping(features_groupDis1, features_groupDis2, T, k_eigen, clusters, num_iters):
    # print(features_groupDis1.size())
    criterion = torch.nn.CrossEntropyLoss().cuda()
    # K-way normalized cuts or k-Means. Default: k-Means

    cluster_label1, centroids1 = KMeans(features_groupDis1, clusters, num_iters)
    cluster_label2, centroids2 = KMeans(features_groupDis2, clusters, num_iters)

    # group discriminative learning
    affnity1 = torch.mm(features_groupDis1, centroids2.t())
    CLD_loss = criterion(affnity1.div_(T), cluster_label2)

    affnity2 = torch.mm(features_groupDis2, centroids1.t())
    CLD_loss = (CLD_loss + criterion(affnity2.div_(T), cluster_label1))/2
    return CLD_loss

def KMeans(x, K=10, Niters=10, verbose=False):
    N, D = x.shape  # Number of samples, dimension of the ambient space
    c = x[:K, :].clone()  # Simplistic random initialization
    x_i = x[:, None, :]  # (Npoints, 1, D)

    for i in range(Niters):
        c_j = c[None, :, :]  # (1, Nclusters, D)
        # c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster
        Ncl = cl.view(cl.size(0), 1).expand(-1, D)
        unique_labels, labels_count = Ncl.unique(dim=0, return_counts=True)
        # As some clusters don't contain any samples, manually assign count as 1
        labels_count_all = torch.ones([K]).long().cuda()
        labels_count_all[unique_labels[:,0]] = labels_count
        c = torch.zeros([K, D], dtype=torch.float).cuda().scatter_add_(0, Ncl, x)
        c = c / labels_count_all.float().unsqueeze(1)

    return cl, c

# def Info_NCE(x, opt, T =1):
#     fix_weight = np.ones((opt.batchSize, opt.batchSize)) - np.identity(opt.batchSize)
#     fix_weight_loss = np.tile(fix_weight, (1, 2))
#     fix_weight_loss = torch.from_numpy(fix_weight_loss).type(torch.FloatTensor).cuda()
    
#     l_pos = torch.einsum('nc,nc->n', [x[(0) * opt.batchSize:(0+1) * opt.batchSize, :], x[(1) * opt.batchSize:(1+1) * opt.batchSize, :]]).unsqueeze(-1)
#     l_neg1 = torch.einsum('nc,ck->nk', [x[(0) * opt.batchSize:(0+1) * opt.batchSize, :], x.permute(1, 0)])
#     l_neg2 = torch.einsum('nc,ck->nk', [x[(1) * opt.batchSize:(1+1) * opt.batchSize, :], x.permute(1, 0)])

#     l_neg1 = torch.mul(l_neg1, fix_weight_loss)
#     l_neg2 = torch.mul(l_neg2, fix_weight_loss)
#     logits_p= torch.cat([l_pos, l_neg1, l_neg2], dim=1)
#     labels = torch.zeros(logits_p.shape[0], dtype=torch.long).cuda()
#     return logits_p/T, labels

def Info_NCE(x, opt, T =1):
    batchSize = int(x.shape[0]/2)
    fix_weight = np.ones((batchSize, batchSize)) - np.identity(batchSize)
    fix_weight_loss = np.tile(fix_weight, (1, 2))
    fix_weight_loss = torch.from_numpy(fix_weight_loss).type(torch.FloatTensor).cuda()
    
    l_pos = torch.einsum('nc,nc->n', [x[(0) * batchSize:(0+1) * batchSize, :], x[(1) * batchSize:(1+1) * batchSize, :]]).unsqueeze(-1)
    l_neg1 = torch.einsum('nc,ck->nk', [x[(0) * batchSize:(0+1) * batchSize, :], x.permute(1, 0)])
    l_neg2 = torch.einsum('nc,ck->nk', [x[(1) * batchSize:(1+1) * batchSize, :], x.permute(1, 0)])

    l_neg1 = torch.mul(l_neg1, fix_weight_loss)
    l_neg2 = torch.mul(l_neg2, fix_weight_loss)
    logits_p= torch.cat([l_pos, l_neg1, l_neg2], dim=1)
    labels = torch.zeros(logits_p.shape[0], dtype=torch.long).cuda()
    return logits_p/T, labels



def group_points(points, opt):
    # group points using knn and ball query
    # points: B * SAMPLE_NUM * 8
    opt.knn_K = 64
    opt.ball_radius = 0.14
    cur_train_size = points.shape[0]
    opt.INPUT_FEATURE_NUM = points.shape[-1]

    points = points.view(cur_train_size, opt.SAMPLE_NUM, -1)
    #print(points.shape)
    inputs1_diff = points[:,:,0:3].transpose(1,2).unsqueeze(1).expand(cur_train_size,opt.sample_num_level1,3,opt.SAMPLE_NUM) \
                 - points[:,0:opt.sample_num_level1,0:3].unsqueeze(-1).expand(cur_train_size,opt.sample_num_level1,3,opt.SAMPLE_NUM)# B * 512 * 3 * 1024
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 512 * 3 * 1024
    inputs1_diff = inputs1_diff.sum(2)                      # B * 512 * 1024 distance
    dists, inputs1_idx = torch.topk(inputs1_diff, opt.knn_K, 2, largest=False, sorted=False)  # dists: B * 512 * 64; inputs1_idx: B * 512 * 64
        
    # ball query
    invalid_map = dists.gt(opt.ball_radius) # B * 512 * 32  value: binary

    for jj in range(opt.sample_num_level1):
        inputs1_idx[:,jj,:][invalid_map[:,jj,:]] = jj
        
    idx_group_l1_long = inputs1_idx.view(cur_train_size,opt.sample_num_level1*opt.knn_K,1).expand(cur_train_size,opt.sample_num_level1*opt.knn_K,opt.INPUT_FEATURE_NUM)

    inputs_level1 = points.gather(1,idx_group_l1_long).view(cur_train_size,opt.sample_num_level1,opt.knn_K,opt.INPUT_FEATURE_NUM) # B*512*64*4

    inputs_level1_center = points[:,0:opt.sample_num_level1,0:3].unsqueeze(2)       # B*512*1*3
    inputs_level1[:,:,:,0:3] = inputs_level1[:,:,:,0:3] - inputs_level1_center.expand(cur_train_size,opt.sample_num_level1,opt.knn_K,3)
    inputs_level1 = inputs_level1.unsqueeze(1).transpose(1,4).squeeze(4)  # B*4*512*64
    inputs_level1_center = inputs_level1_center.contiguous().view(-1,1,opt.sample_num_level1,3).transpose(1,3)  # B*3*512*1

    ##
    # inputs_level1 = inputs_level1.view(-1,opt.temperal_num,opt.INPUT_FEATURE_NUM,opt.sample_num_level1,opt.knn_K)
    # inputs_level1_center = inputs_level1_center.view(-1,opt.temperal_num,3,opt.sample_num_level1,1)

    return inputs_level1, inputs_level1_center
    #inputs_level1: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, inputs_level1_center: B*3*sample_num_level1*1

def group_points_3DV(points, opt):
    # group points using knn and ball query
    # points: B * SAMPLE_NUM * 4
    cur_train_size = points.shape[0]
    opt.INPUT_FEATURE_NUM = points.shape[-1]
    # opt.knn_K = 64
    opt.SAMPLE_NUM = points.shape[1]
    opt.ball_radius = 0.1

    points = points.view(cur_train_size, opt.SAMPLE_NUM, -1)
    #print(points.shape)
    inputs1_diff = points[:,:,0:3].transpose(1,2).unsqueeze(1).expand(cur_train_size,opt.sample_num_level1,3,opt.SAMPLE_NUM) \
                 - points[:,0:opt.sample_num_level1,0:3].unsqueeze(-1).expand(cur_train_size,opt.sample_num_level1,3,opt.SAMPLE_NUM)# B * 512 * 3 * 1024
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 128 * 3 * 512
    inputs1_diff = inputs1_diff.sum(2)                      # B * 128 * 512 distance
    dists, inputs1_idx = torch.topk(inputs1_diff, opt.knn_K, 2, largest=False, sorted=False)  # dists: B * 128 * 32; inputs1_idx: B * 128 * 32
        
    # ball query
    invalid_map = dists.gt(opt.ball_radius) # B * 128 * 32  value: binary

    for jj in range(opt.sample_num_level1):
        inputs1_idx[:,jj,:][invalid_map[:,jj,:]] = jj
        
    idx_group_l1_long = inputs1_idx.view(cur_train_size,opt.sample_num_level1*opt.knn_K,1).expand(cur_train_size,opt.sample_num_level1*opt.knn_K,opt.INPUT_FEATURE_NUM)
    #print(points.shape,'points')
    inputs_level1 = points.gather(1,idx_group_l1_long).view(cur_train_size,opt.sample_num_level1,opt.knn_K,opt.INPUT_FEATURE_NUM) # B*128*32*4

    inputs_level1_center = points[:,0:opt.sample_num_level1,0:3].unsqueeze(2)       # B*128*1*3
    inputs_level1[:,:,:,0:3] = inputs_level1[:,:,:,0:3] - inputs_level1_center.expand(cur_train_size,opt.sample_num_level1,opt.knn_K,inputs_level1_center.shape[-1])
    inputs_level1[:,:,:,0:3] = inputs_level1[:,:,:,0:3]#/opt.ball_radius
    
    inputs_level1 = inputs_level1.unsqueeze(1).transpose(1,4).squeeze(4)  # B*4*128*32
    inputs_level1_center = inputs_level1_center.contiguous().view(-1,1,opt.sample_num_level1,3).transpose(1,3)  # B*3*128*1

    ##
    # inputs_level1 = inputs_level1.view(-1,opt.temperal_num,opt.INPUT_FEATURE_NUM,opt.sample_num_level1,opt.knn_K)
    # inputs_level1_center = inputs_level1_center.view(-1,opt.temperal_num,3,opt.sample_num_level1,1)

    return inputs_level1, inputs_level1_center
    #inputs_level1: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, inputs_level1_center: B*3*sample_num_level1*1

def group_points_3DV_nums(points, sample_num_level1=32, knn_K=64):
    # group points using knn and ball query
    # points: B * SAMPLE_NUM * 4
    cur_train_size = points.shape[0]
    INPUT_FEATURE_NUM = points.shape[-1]
    SAMPLE_NUM = points.shape[1]
    # knn_K = 64
    ball_radius = 0.2

    points = points.view(cur_train_size, SAMPLE_NUM, -1)
    #print(points.shape)
    inputs1_diff = points[:,:,0:3].transpose(1,2).unsqueeze(1).expand(cur_train_size,sample_num_level1,3,SAMPLE_NUM) \
                 - points[:,0:sample_num_level1,0:3].unsqueeze(-1).expand(cur_train_size,sample_num_level1,3,SAMPLE_NUM)# B * 512 * 3 * 1024
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 128 * 3 * 512
    inputs1_diff = inputs1_diff.sum(2)                      # B * 128 * 512 distance
    dists, inputs1_idx = torch.topk(inputs1_diff, knn_K, 2, largest=False, sorted=False)  # dists: B * 128 * 32; inputs1_idx: B * 128 * 32
        
    # ball query
    invalid_map = dists.gt(ball_radius) # B * 128 * 32  value: binary

    for jj in range(sample_num_level1):
        inputs1_idx[:,jj,:][invalid_map[:,jj,:]] = jj
        
    idx_group_l1_long = inputs1_idx.view(cur_train_size,sample_num_level1*knn_K,1).expand(cur_train_size,sample_num_level1*knn_K,INPUT_FEATURE_NUM)
    #print(points.shape,'points')
    inputs_level1 = points.gather(1,idx_group_l1_long).view(cur_train_size,sample_num_level1,knn_K,INPUT_FEATURE_NUM) # B*128*32*4

    inputs_level1_center = points[:,0:sample_num_level1,0:3].unsqueeze(2)       # B*128*1*3
    inputs_level1[:,:,:,0:3] = inputs_level1[:,:,:,0:3] - inputs_level1_center.expand(cur_train_size,sample_num_level1,knn_K,inputs_level1_center.shape[-1])
    inputs_level1 = inputs_level1.unsqueeze(1).transpose(1,4).squeeze(4)  # B*4*128*32
    inputs_level1_center = inputs_level1_center.contiguous().view(-1,1,sample_num_level1,3).transpose(1,3)  # B*3*128*1

    return inputs_level1, inputs_level1_center


def group_points_3DV_nums_set(points, sample_num_level1=64, knn_K=64):
    # group points using knn and ball query
    # points: B * SAMPLE_NUM * 4
    cur_train_size = points.shape[0]
    INPUT_FEATURE_NUM = points.shape[-1]
    SAMPLE_NUM = points.shape[1]
    # knn_K = 64
    ball_radius = 0.2

    points = points.view(cur_train_size, SAMPLE_NUM, -1)
    #print(points.shape)
    inputs1_diff = points[:,:,0:3].transpose(1,2).unsqueeze(1).expand(cur_train_size,sample_num_level1,3,SAMPLE_NUM) \
                 - points[:,0:sample_num_level1,0:3].unsqueeze(-1).expand(cur_train_size,sample_num_level1,3,SAMPLE_NUM)# B * 512 * 3 * 1024
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 128 * 3 * 512
    inputs1_diff = inputs1_diff.sum(2)                      # B * 128 * 512 distance
    dists, inputs1_idx = torch.topk(inputs1_diff, knn_K, 2, largest=False, sorted=False)  # dists: B * 128 * 32; inputs1_idx: B * 128 * 32
        
    # ball query
    invalid_map = dists.gt(ball_radius) # B * 128 * 32  value: binary

    for jj in range(sample_num_level1):
        inputs1_idx[:,jj,:][invalid_map[:,jj,:]] = jj
        
    idx_group_l1_long = inputs1_idx.view(cur_train_size,sample_num_level1*knn_K,1).expand(cur_train_size,sample_num_level1*knn_K,INPUT_FEATURE_NUM)
    #print(points.shape,'points')
    inputs_level1 = points.gather(1,idx_group_l1_long).view(cur_train_size,sample_num_level1,knn_K,INPUT_FEATURE_NUM) # B*128*32*4

    inputs_level1_center = points[:,0:sample_num_level1,0:3].unsqueeze(2)       # B*128*1*3
    
    inputs_level1_no_center = inputs_level1.clone()
    inputs_level1[:,:,:,0:3] = inputs_level1[:,:,:,0:3] - inputs_level1_center.expand(cur_train_size,sample_num_level1,knn_K,inputs_level1_center.shape[-1])
    
    inputs_level1 = inputs_level1.unsqueeze(1).transpose(1,4).squeeze(4)  # B*4*128*32
    inputs_level1_center = inputs_level1_center.contiguous().view(-1,1,sample_num_level1,3).transpose(1,3)  # B*3*128*1
    
    inputs_level1_no_center = inputs_level1_no_center.reshape(inputs_level1_no_center.shape[0], -1, 4)
    inputs_level1_no_center = inputs_level1_no_center[:, :512, :]

    return inputs_level1, inputs_level1_center, inputs_level1_no_center



def group_points_2(points, sample_num_level1, sample_num_level2, knn_K, ball_radius):
    # group points using knn and ball query
    # points: B*(3+128)*512
    knn_K = torch.tensor(64)
    #ball_radius = 0.26
    cur_train_size = points.size(0)
    inputs1_diff = points[:,0:3,:].unsqueeze(1).expand(cur_train_size,sample_num_level2,3,sample_num_level1) \
                 - points[:,0:3,0:sample_num_level2].transpose(1,2).unsqueeze(-1).expand(cur_train_size,sample_num_level2,3,sample_num_level1)# B * 128 * 3 * 512
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 128 * 3 * 512
    inputs1_diff = inputs1_diff.sum(2)                      # B * 128 * 512
    dists, inputs1_idx = torch.topk(inputs1_diff, knn_K, 2, largest=False, sorted=False)  # dists: B * 128 * 64; inputs1_idx: B * 128 * 64
        
    # ball query
    invalid_map = dists.gt(ball_radius.cuda()) # B * 128 * 64, invalid_map.float().sum()

    for jj in range(sample_num_level2):
        inputs1_idx.data[:,jj,:][invalid_map.data[:,jj,:]] = jj

    idx_group_l1_long = inputs1_idx.view(cur_train_size,1,sample_num_level2*knn_K).expand(cur_train_size,points.size(1),sample_num_level2*knn_K)
    inputs_level2 = points.gather(2,idx_group_l1_long).view(cur_train_size,points.size(1),sample_num_level2,knn_K) # B*131*128*64

    inputs_level2_center = points[:,0:3,0:sample_num_level2].unsqueeze(3)       # B*3*128*1
    inputs_level2[:,0:3,:,:] = inputs_level2[:,0:3,:,:] - inputs_level2_center.expand(cur_train_size,3,sample_num_level2,knn_K) # B*3*128*64
    return inputs_level2, inputs_level2_center
    # inputs_level2: B*131*sample_num_level2*knn_K, inputs_level2_center: B*3*sample_num_level2*1

def group_points_2_3DV(points, sample_num_level1, sample_num_level2, knn_K, ball_radius):
    # group points using knn and ball query
    # points: B*(3+128)*512
    knn_K = torch.tensor(64)
    ball_radius = torch.tensor(0.11)
    cur_train_size = points.size(0)
    inputs1_diff = points[:,0:3,:].unsqueeze(1).expand(cur_train_size,sample_num_level2,3,sample_num_level1) \
                 - points[:,0:3,0:sample_num_level2].transpose(1,2).unsqueeze(-1).expand(cur_train_size,sample_num_level2,3,sample_num_level1)# B * 128 * 3 * 512
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 128 * 3 * 512
    inputs1_diff = inputs1_diff.sum(2)                      # B * 128 * 512
    dists, inputs1_idx = torch.topk(inputs1_diff, knn_K, 2, largest=False, sorted=False)  # dists: B * 128 * 64; inputs1_idx: B * 128 * 64
        
    # ball query
    invalid_map = dists.gt(ball_radius.cuda()) # B * 128 * 64, invalid_map.float().sum()

    for jj in range(sample_num_level2):
        inputs1_idx.data[:,jj,:][invalid_map.data[:,jj,:]] = jj

    idx_group_l1_long = inputs1_idx.view(cur_train_size,1,sample_num_level2*knn_K).expand(cur_train_size,points.size(1),sample_num_level2*knn_K)
    inputs_level2 = points.gather(2,idx_group_l1_long).view(cur_train_size,points.size(1),sample_num_level2,knn_K) # B*131*128*64

    inputs_level2_center = points[:,0:3,0:sample_num_level2].unsqueeze(3)       # B*8*128*1
    inputs_level2[:,0:3,:,:] = inputs_level2[:,0:3,:,:] - inputs_level2_center.expand(cur_train_size,3,sample_num_level2,knn_K) # B*8*128*64
    return inputs_level2, inputs_level2_center
    # inputs_level2: B*136*sample_num_level2*knn_K; inputs_level2_center: B*8*sample_num_level2*1
    
    
def group_points_2_fetaure(points, sample_num_level1 =16, sample_num_level2 = 16, knn_K= 4):
        # group points using knn and ball query
    # points: B*(3+128)*512
    channel = points.shape[1]
    knn_K = torch.tensor(4)
    ball_radius = torch.tensor(10000)
    cur_train_size = points.size(0)
    inputs1_diff = points[:,:,:].unsqueeze(1).expand(cur_train_size,sample_num_level2,channel,sample_num_level1) \
                 - points[:,:,0:sample_num_level2].transpose(1,2).unsqueeze(-1).expand(cur_train_size,sample_num_level2,channel,sample_num_level1)# B * 128 * 3 * 512
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 128 * 3 * 512
    inputs1_diff = inputs1_diff.sum(2)                      # B * 128 * 512
    dists, inputs1_idx = torch.topk(inputs1_diff, knn_K, 2, largest=False, sorted=False)  # dists: B * 128 * 64; inputs1_idx: B * 128 * 64
        
    # ball query
    invalid_map = dists.gt(ball_radius.cuda()) # B * 128 * 64, invalid_map.float().sum()

    for jj in range(sample_num_level2):
        inputs1_idx.data[:,jj,:][invalid_map.data[:,jj,:]] = jj

    idx_group_l1_long = inputs1_idx.view(cur_train_size,1,sample_num_level2*knn_K).expand(cur_train_size,points.size(1),sample_num_level2*knn_K)
    inputs_level2 = points.gather(2,idx_group_l1_long).view(cur_train_size,points.size(1),sample_num_level2,knn_K) # B*131*128*64

    inputs_level2_center = points[:,:,0:sample_num_level2].unsqueeze(3)       # B*8*128*1
    inputs_level2[:,:,:,:] = inputs_level2[:,:,:,:] - inputs_level2_center.expand(cur_train_size,channel,sample_num_level2,knn_K) # B*8*128*64
    return inputs_level2, inputs_level2_center


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



def cluster_loss(code, batchSize, num_crop = 2, queue = None):
    loss_swa = 0
    for crop_id in range(num_crop-1):  # 4 kinds positive samples  crop_id is the all kinds of augmentation data
        with torch.no_grad():
            po = code[batchSize * crop_id:batchSize * (crop_id + 1), :]  # batch * centers
            # calculate the queue to the code according to the properties
            if queue is not None:
                if use_the_queue or not torch.all(queue[crop_id, -1, :] == 0):
                    use_the_queue = True
                    # aa= torch.mm(queue[i], netR.module.mapping.weight.t())
                    po = torch.cat((torch.mm(queue[crop_id], netR.module.mapping.weight.t()), po))
                # fill the queue
                queue[crop_id, batchSize:, :] = queue[crop_id, :-batchSize, :].clone()
                queue[crop_id, 0:batchSize, :] = x_nor[crop_id * batchSize: (crop_id + 1) * batchSize, :]

            po = po / 0.03
            aa = po.detach().cpu().numpy()
            po = torch.exp(po).t()  # centers * batch
            q = distributed_sinkhorn(po, 3)[-batchSize:]
        subloss = 0
        for v in np.delete(np.arange(num_crop-1), crop_id):
            p = softmax(code[batchSize * v: batchSize * (v + 1)] / 0.1)
            subloss = subloss - torch.mean(torch.sum(q * torch.log(p), dim=1))
            ss = subloss.detach().cpu().numpy()
        loss_swa = loss_swa + subloss
    loss_swa = loss_swa / (num_crop-1)
    return loss_swa