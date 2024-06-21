 #!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Liner(nn.Module):
    def __init__(self, opt, num_class=120, gost=3, dim=512, normalize_input=True):
        super(Liner, self).__init__()
       
        self.fc =  nn.Linear(dim*gost, num_class)
    
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()
        

    def forward(self, xt):
        B, D = xt.shape
        # xt = xt.reshape(-1, 512*1)
        xt = F.normalize(xt, p=2, dim=1)
        x_nor = xt.reshape(-1, D)
        presdiction = self.fc(x_nor)
        # presdiction = self.ac(presdiction)
        # presdiction = self.fc1(presdiction)

        return presdiction







