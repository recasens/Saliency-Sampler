import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
import numpy as np
import random



def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


class Saliency_Sampler(nn.Module):
    def __init__(self,task_network,saliency_network,task_input_size,saliency_input_size):
        super(Saliency_Sampler, self).__init__()
        
        self.hi_res = task_network
        self.grid_size = 31
        self.padding_size = 30
        self.global_size = self.grid_size+2*self.padding_size
        self.input_size = saliency_input_size
        self.input_size_net = task_input_size
        self.conv_last = nn.Conv2d(256,1,kernel_size=1,padding=0,stride=1)
        gaussian_weights = torch.FloatTensor(makeGaussian(2*self.padding_size+1, fwhm = 13))

        # Spatial transformer localization-network
        self.localization = saliency_network
        self.filter = nn.Conv2d(1, 1, kernel_size=(2*self.padding_size+1,2*self.padding_size+1),bias=False)
        self.filter.weight[0].data[:,:,:] = gaussian_weights

        self.P_basis = torch.zeros(2,self.grid_size+2*self.padding_size, self.grid_size+2*self.padding_size)
        for k in range(2):
            for i in range(self.global_size):
                for j in range(self.global_size):
                    self.P_basis[k,i,j] = k*(i-self.padding_size)/(self.grid_size-1.0)+(1.0-k)*(j-self.padding_size)/(self.grid_size-1.0)

    def create_grid(self, x):
        P = torch.autograd.Variable(torch.zeros(1,2,self.grid_size+2*self.padding_size, self.grid_size+2*self.padding_size).cuda(),requires_grad=False)
        P[0,:,:,:] = self.P_basis
        P = P.expand(x.size(0),2,self.grid_size+2*self.padding_size, self.grid_size+2*self.padding_size)


        x_cat = torch.cat((x,x),1)
        p_filter = self.filter(x)
        x_mul = torch.mul(P,x_cat).view(-1,1,self.global_size,self.global_size)
        all_filter = self.filter(x_mul).view(-1,2,self.grid_size,self.grid_size)


        x_filter = all_filter[:,0,:,:].contiguous().view(-1,1,self.grid_size,self.grid_size)
        y_filter = all_filter[:,1,:,:].contiguous().view(-1,1,self.grid_size,self.grid_size)

        x_filter = x_filter/p_filter
        y_filter = y_filter/p_filter

        xgrids = x_filter*2-1
        ygrids = y_filter*2-1
        xgrids = torch.clamp(xgrids,min=-1,max=1)
        ygrids = torch.clamp(ygrids,min=-1,max=1)


        xgrids = xgrids.view(-1,1,self.grid_size,self.grid_size)
        ygrids = ygrids.view(-1,1,self.grid_size,self.grid_size)

        grid = torch.cat((xgrids,ygrids),1)

        grid = nn.Upsample(size=(self.input_size_net,self.input_size_net), mode='bilinear')(grid)

        grid = torch.transpose(grid,1,2)
        grid = torch.transpose(grid,2,3)

        return grid

    def forward(self, x,p):

        x_low = nn.AdaptiveAvgPool2d((self.input_size,self.input_size))(x)
        xs = self.localization(x_low)
        xs = nn.ReLU()(xs)
        xs = self.conv_last(xs)
        xs = nn.Upsample(size=(self.grid_size,self.grid_size), mode='bilinear')(xs)
        xs = xs.view(-1,self.grid_size*self.grid_size)
        xs = nn.Softmax()(xs)
        xs = xs.view(-1,1,self.grid_size,self.grid_size)
        xs_hm = nn.ReplicationPad2d(self.padding_size)(xs)

        grid = self.create_grid(xs_hm)

        x_sampled = F.grid_sample(x, grid)

        if random.random()>p:
            s = random.randint(64, 224)
            x_sampled = nn.AdaptiveAvgPool2d((s,s))(x_sampled)
            x_sampled = nn.Upsample(size=(self.input_size_net,self.input_size_net),mode='bilinear')(x_sampled)

        x = self.hi_res(x_sampled)

        return x,x_sampled,xs



