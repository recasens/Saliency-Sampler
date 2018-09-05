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
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models

from saliency_network import saliency_network_resnet18
import numpy as np
from resnet import resnet101
import random
from saliency_sampler import Saliency_Sampler

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

network_name = 'imagenet'
from tensorboardX import SummaryWriter
foo = SummaryWriter(comment=network_name)


epochs = 120
N_pretraining = 30

momentum = 0.9
weight_decay = 1e-4
best_prec1 = 0

doLoad = False
lr = 0.1
base_lr = 0.1

batch_size = 256

workers = 10
data_path = '/data/vision/torralba/datasets/imagenet_pytorch'

count = 0
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
best_prec1 = 0


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




def main():
    global best_prec1,count

    model_v = Saliency_Sampler(resnet101(),saliency_network_resnet18(),224,224)
    model = torch.nn.DataParallel(model_v).cuda()
    model.cuda()

    if doLoad:
        checkpoint = torch.load('checkpoint_'+network_name+'.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        epoch_init = checkpoint['epoch']
        count = 5201*epoch_init



    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD([
                {'params': model.module.hi_res.parameters(),'lr_mult':1,'zoom':False},
                {'params': model.module.conv_last.parameters(),'lr_mult':0.01,'zoom':True},
                {'params': model.module.localization.parameters(), 'lr_mult': 0.001,'zoom':True}
                ],lr =base_lr,momentum=momentum,weight_decay=weight_decay)



    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(800),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize((800,800)),
            transforms.CenterCrop(700),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)


    for epoch in range(epoch_init, epochs):
        adjust_learning_rate(optimizer,epoch)
        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    global count,unorm
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        if epoch>N_pretraining:
            p=1
        else:
            p=0

        # compute output
        output,image_output,hm = model(input_var,p)
        loss = criterion(output, target_var)

        image_output = image_output.data
        image_output = image_output[0,:,:,:]
        image_output = unorm(image_output)

        hm = hm.data[0,:,:,:]
        hm = hm/hm.view(-1).max()


        xhm = vutils.make_grid(hm, normalize=True, scale_each=True)
        foo.add_image('Heat Map', xhm, count)


        x = vutils.make_grid(image_output, normalize=True, scale_each=True)
        foo.add_image('Image', x, count)


        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        foo.add_scalar("data/top1", top1.val, count)
        foo.add_scalar("data/top5", top5.val, count)
        foo.add_scalar("data/loss", losses.val, count)


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        count = count +1

        print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
             epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    global count,unorm

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        with torch.no_grad():
            # compute output
            output,image_output,hm = model(input_var,1)
            loss = criterion(output, target_var)

            image_output = image_output.data
            image_output = image_output[0,:,:,:]
            image_output = unorm(image_output)

            hm = hm.data[0,:,:,:]
            hm = hm/hm.view(-1).max()

            xhm = vutils.make_grid(hm, normalize=True, scale_each=True)
            foo.add_image('Heat Map-Test', xhm, count)


            x = vutils.make_grid(image_output, normalize=True, scale_each=True)
            foo.add_image('Image Test', x, count)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            count = count +1

            print('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    foo.add_scalar("data/top1-test", top1.avg, count)
    foo.add_scalar("data/top5-test", top5.avg, count)


    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint_'+network_name+'.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_'+network_name+'.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer,epoch):

    if epoch>N_pretraining:
        lr_class = base_lr*0.1**((epoch-N_pretraining)//30)
        lr_zoom = base_lr*0.1**(epoch//30)
    else:
        lr_class = base_lr*0.1**(epoch//30)
        lr_zoom = base_lr*0.1**(epoch//30)

    
    for param_group in optimizer.param_groups:
        if param_group['zoom']==True:
            param_group['lr'] = param_group['lr_mult']*lr_zoom
        else:
            param_group['lr'] = param_group['lr_mult']*lr_class


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

