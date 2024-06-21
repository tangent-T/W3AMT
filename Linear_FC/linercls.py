import logging

import torch
import os
import tqdm
import shutil
import collections
import argparse
import random
import time
# import gpu_utils as g
import numpy as np

# *******************************mode change to use dgcnn
import model as MODELL
from data_set import LIner_NTU

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


id = 2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(args=None):
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--dataset', type=str, default='ntu120',help='how to aggregate temporal split features: ntu120 | ntu60')

    parser.add_argument('--weight_decay', type=float, default=0.0008, help='weight decay (SGD only)')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate at t=0')   # 0.005效果最好
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (SGD only)')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')

    parser.add_argument('--root_path', type=str,default='../data/raw/',help='preprocess folder')

    parser.add_argument('--save_root_dir', type=str, default='../data/models/',help='output folder')
    parser.add_argument('--optimizer', type=str, default='', help='optimizer name for training resume')

    parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
    parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id')  # CUDA_VISIBLE_DEVICES=0 python train.py

    parser.add_argument('--learning_rate_decay', type=float, default=1e-7, help='learning rate decay')
 

    opt = parser.parse_args()
    print(opt)

    torch.cuda.set_device(opt.main_gpu)

    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)


    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    
    root_path_ = '../data/raw/'
    data_train = LIner_NTU(root_path=root_path_, opt=opt,
                          # opt.root_path
                          DATA_CROSS_VIEW=True,
                          full_train=True,
                          validation=False,
                          test=False,
                          DATA_CROSS_SET=False
                          )

    # get all the data fold into batches
    train_loader = DataLoader(dataset=data_train, batch_size=opt.batchSize, shuffle=True, drop_last=True, num_workers=8)


    data_val = LIner_NTU(root_path=root_path_, opt=opt,
                          # opt.root_path
                          DATA_CROSS_VIEW=True,
                          full_train=False,
                          validation=False,
                          test=True,
                          DATA_CROSS_SET=False
                          )
    test_loader = DataLoader(dataset=data_val, batch_size=opt.batchSize, drop_last=False, shuffle=False, num_workers=8)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    netR = MODELL.Liner(opt)
    netR = torch.nn.DataParallel(netR).cuda()
    netR.cuda()

    optimizer = torch.optim.Adam(netR.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999), eps=1e-06)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    best = 0

    torch.cuda.synchronize()
    # print('now start extrat train data fetaure... ')
    best_acc = 0
    for epoch in range(0, opt.nepoch):
        conf_mat = np.zeros([120, 120])
        # switch to train mode
        torch.cuda.synchronize()
        netR.train()
        loss_sigma = 0.0

        # need to rewite data_get
        for i, data in enumerate(tqdm(train_loader, 0)):
            # print(i)
            if len(data[0]) == 1:
                continue
            feature, target, _ = data
            data1 = feature.type(torch.FloatTensor)
            data1 = data1  # 5* batch  * 512 * 4
            data1 = data1.cuda()
            label = target.cuda()
            
            torch.cuda.synchronize()

            prediction= netR(data1)
            loss = criterion(prediction, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)
            torch.cuda.synchronize()
            loss_sigma += loss.item()

            
            _, predicted = torch.max(prediction.data, 1)
            #print(prediction.data)
            
            for j in range(len(label)):
                cate_i = label[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.0
                
        print('epoch:', epoch, '--ACC :', 100*conf_mat.trace() / conf_mat.sum(), '%--loss:', loss_sigma / (i + 1))
        
        with torch.no_grad():
            if epoch%2==0:
                if epoch>5:
                    netR.eval()
                    # need to rewite data_get
                    conf_mat = np.zeros([120,120])
                    for i, data in enumerate(tqdm(test_loader, 0)):
                        # print(i)
                        if len(data[0]) == 1:
                            continue
                        feature, target, _ = data
                        data1 = feature.type(torch.FloatTensor)
                        data1 = data1.cuda()
                        label = target.cuda()
                        
                        torch.cuda.synchronize()

                        prediction= netR(data1)
                        loss = criterion(prediction, label)
                
                        _, predicted = torch.max(prediction.data, 1)
                        #print(prediction.data)
                        
                        for j in range(len(label)):
                            cate_i = label[j].cpu().numpy()
                            pre_i = predicted[j].cpu().numpy()
                            conf_mat[cate_i, pre_i] += 1.0
                            
                    print('epoch:', epoch, '--ACC :', 100*conf_mat.trace() / conf_mat.sum())
                    now_acc = 100*conf_mat.trace() / conf_mat.sum()
                    if now_acc>best_acc:
                        best_acc = now_acc
                        torch.save(netR.module.state_dict(), '%s/ntu_cv_best.pth' % ('/data/data1/'))

    

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

if __name__ == '__main__':
    main()