##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

import encoding
from encoding.utils import (accuracy, AverageMeter, MixUpWrapper, LR_Scheduler)

class Options():
    def __init__(self):
        # data settings
        parser = argparse.ArgumentParser(description='Deep Encoding')
        parser.add_argument('--dataset', type=str, default='imagenet',
                            help='training dataset (default: imagenet)')
        parser.add_argument('--base-size', type=int, default=None,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=224,
                            help='crop image size')
        # model params 
        parser.add_argument('--model', type=str, default='densenet',
                            help='network model type (default: densenet)')
        parser.add_argument('--rectify', action='store_true', 
                            default=False, help='rectify convolution')
        parser.add_argument('--rectify-avg', action='store_true', 
                            default=False, help='rectify convolution')
        # training hyper params
        parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                            help='batch size for training (default: 128)')
        parser.add_argument('--workers', type=int, default=32,
                            metavar='N', help='dataloader threads')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', 
                            default=False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--verify', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--export', type=str, default=None,
                            help='put the path to resuming file if needed')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args


def main():
    # init the args
    args = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # init dataloader
    _, transform_val = encoding.transforms.get_transform(args.dataset, args.base_size, args.crop_size)
    valset = encoding.datasets.get_dataset(args.dataset, root=os.path.expanduser('~/.encoding/data'),
                                           transform=transform_val, train=False, download=True)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True if args.cuda else False)
    
    # init the model
    model_kwargs = {'pretrained': True}

    if args.rectify:
        model_kwargs['rectified_conv'] = True
        model_kwargs['rectify_avg'] = args.rectify_avg

    model = encoding.models.get_model(args.model, **model_kwargs)
    print(model)

    if args.cuda:
        model.cuda()
        # Please use CUDA_VISIBLE_DEVICES to control the number of gpus
        model = nn.DataParallel(model)

    # checkpoint
    if args.verify:
        if os.path.isfile(args.verify):
            print("=> loading checkpoint '{}'".format(args.verify))
            model.module.load_state_dict(torch.load(args.verify))
        else:
            raise RuntimeError ("=> no verify checkpoint found at '{}'".\
                format(args.verify))
    elif args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            raise RuntimeError ("=> no resume checkpoint found at '{}'".\
                format(args.resume))

    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    is_best = False
    tbar = tqdm(val_loader, desc='\r')
    for batch_idx, (data, target) in enumerate(tbar):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

        tbar.set_description('Top1: %.3f | Top5: %.3f'%(top1.avg, top5.avg))

    print('Top1 Acc: %.3f | Top5 Acc: %.3f '%(top1.avg, top5.avg))

    if args.export:
        torch.save(model.module.state_dict(), args.export + '.pth')


if __name__ == "__main__":
    main()

