##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import print_function
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

import encoding
from encoding.nn import LabelSmoothing, NLLMultiLabelSmooth
from encoding.utils import (accuracy, AverageMeter, MixUpWrapper, LR_Scheduler)

class Options():
    def __init__(self):
        # data settings
        parser = argparse.ArgumentParser(description='Deep Encoding')
        parser.add_argument('--dataset', type=str, default='cifar10',
                            help='training dataset (default: cifar10)')
        parser.add_argument('--base-size', type=int, default=None,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=224,
                            help='crop image size')
        parser.add_argument('--label-smoothing', type=float, default=0.0,
                            help='label-smoothing (default eta: 0.0)')
        parser.add_argument('--mixup', type=float, default=0.0,
                            help='mixup (default eta: 0.0)')
        parser.add_argument('--rand-aug', action='store_true', 
                            default=False, help='rectify convolution')
        # model params 
        parser.add_argument('--model', type=str, default='densenet',
                            help='network model type (default: densenet)')
        parser.add_argument('--pretrained', action='store_true', 
                            default=False, help='load pretrianed mode')
        parser.add_argument('--rectify', action='store_true', 
                            default=False, help='rectify convolution')
        parser.add_argument('--rectify-avg', action='store_true', 
                            default=False, help='rectify convolution')
        parser.add_argument('--last-gamma', action='store_true', default=False,
                            help='whether to init gamma of the last BN layer in \
                            each bottleneck to 0 (default: False)')
        parser.add_argument('--dropblock-prob', type=float, default=0,
                            help='DropBlock prob. default is 0.')
        parser.add_argument('--final-drop', type=float, default=0,
                            help='final dropout prob. default is 0.')
        # training hyper params
        parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                            help='batch size for training (default: 128)')
        parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                            help='batch size for testing (default: 256)')
        parser.add_argument('--epochs', type=int, default=120, metavar='N',
                            help='number of epochs to train (default: 600)')
        parser.add_argument('--start_epoch', type=int, default=0, 
                            metavar='N', help='the epoch number to start (default: 1)')
        parser.add_argument('--workers', type=int, default=32,
                            metavar='N', help='dataloader threads')
        # optimizer
        parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                            help='learning rate (default: 0.1)')
        parser.add_argument('--lr-scheduler', type=str, default='cos', 
                            help='learning rate scheduler (default: cos)')
        parser.add_argument('--warmup-epochs', type=int, default=0,
                            help='number of warmup epochs (default: 0)')
        parser.add_argument('--lr-step', type=int, default=40, metavar='LR',
                            help='learning rate step (default: 40)')
        parser.add_argument('--momentum', type=float, default=0.9, 
                            metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=1e-4, 
                            metavar ='M', help='SGD weight decay (default: 1e-4)')
        parser.add_argument('--no-bn-wd', action='store_true', 
                            default=False, help='no bias decay')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', 
                            default=False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default='default',
                            help='set the checkpoint name')
        # evaluation option
        parser.add_argument('--eval', action='store_true', default= False,
                            help='evaluating')
        parser.add_argument('--export', type=str, default=None,
                            help='put the path to resuming file if needed')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args

# global variable
best_pred = 0.0
acclist_train = []
acclist_val = []

def main():
    # init the args
    global best_pred, acclist_train, acclist_val
    args = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # init dataloader
    transform_train, transform_val = encoding.transforms.get_transform(
            args.dataset, args.base_size, args.crop_size, args.rand_aug)
    trainset = encoding.datasets.get_dataset(args.dataset, root=os.path.expanduser('~/.encoding/data'),
                                             transform=transform_train, train=True, download=True)
    valset = encoding.datasets.get_dataset(args.dataset, root=os.path.expanduser('~/.encoding/data'),
                                           transform=transform_val, train=False, download=True)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    # init the model
    model_kwargs = {}
    if args.pretrained:
        model_kwargs['pretrained'] = True

    if args.final_drop > 0.0:
        model_kwargs['final_drop'] = args.final_drop

    if args.dropblock_prob > 0.0:
        model_kwargs['dropblock_prob'] = args.dropblock_prob

    if args.last_gamma:
        model_kwargs['last_gamma'] = True

    if args.rectify:
        model_kwargs['rectified_conv'] = True
        model_kwargs['rectify_avg'] = args.rectify_avg

    model = encoding.models.get_model(args.model, **model_kwargs)
    if args.dropblock_prob > 0.0:
        from functools import partial
        from encoding.nn import reset_dropblock
        nr_iters = (args.epochs - 2 * args.warmup_epochs) * len(train_loader)
        apply_drop_prob = partial(reset_dropblock, args.warmup_epochs*len(train_loader),
                                  nr_iters, 0.0, args.dropblock_prob)
        model.apply(apply_drop_prob)

    print(model)
    # criterion and optimizer
    if args.mixup > 0:
        train_loader = MixUpWrapper(args.mixup, 1000, train_loader,
                                    list(range(torch.cuda.device_count())))
        criterion = NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0:
        criterion = LabelSmoothing(args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    if args.no_bn_wd:
        parameters = model.named_parameters()
        param_dict = {}
        for k, v in parameters:
            param_dict[k] = v
        bn_params = [v for n, v in param_dict.items() if ('bn' in n or 'bias' in n)]
        rest_params = [v for n, v in param_dict.items() if not ('bn' in n or 'bias' in n)]
        print(" Weight decay NOT applied to BN parameters ")
        print(f'len(parameters): {len(list(model.parameters()))} = {len(bn_params)} + {len(rest_params)}')
        optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0 },
                                     {'params': rest_params, 'weight_decay': args.weight_decay}],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    if args.cuda:
        model.cuda()
        criterion.cuda()
        # Please use CUDA_VISIBLE_DEVICES to control the number of gpus
        model = nn.DataParallel(model)
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1 if args.start_epoch == 1 else args.start_epoch
            best_pred = checkpoint['best_pred']
            acclist_train = checkpoint['acclist_train']
            acclist_val = checkpoint['acclist_val']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            raise RuntimeError ("=> no resume checkpoint found at '{}'".\
                format(args.resume))
    scheduler = LR_Scheduler(args.lr_scheduler,
                             base_lr=args.lr,
                             num_epochs=args.epochs,
                             iters_per_epoch=len(train_loader),
                             warmup_epochs=args.warmup_epochs,
                             lr_step=args.lr_step)
    def train(epoch):
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        global best_pred, acclist_train
        tbar = tqdm(train_loader, desc='\r')
        for batch_idx, (data, target) in enumerate(tbar):
            scheduler(optimizer, batch_idx, epoch, best_pred)
            #criterion.update(batch_idx, epoch)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            acc1 = accuracy(output, target, topk=(1,))
            top1.update(acc1[0], data.size(0))
            losses.update(loss.item(), data.size(0))
            tbar.set_description('\rLoss: %.3f | Top1: %.3f'%(losses.avg, top1.avg))

        acclist_train += [top1.avg]

    def validate(epoch):
        model.eval()
        top1 = AverageMeter()
        top5 = AverageMeter()
        global best_pred, acclist_train, acclist_val
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

        if args.eval:
            print('Top1 Acc: %.3f | Top5 Acc: %.3f '%(top1.avg, top5.avg))
            return
        # save checkpoint
        acclist_val += [top1.avg]
        if top1.avg > best_pred:
            best_pred = top1.avg 
            is_best = True
        encoding.utils.save_checkpoint({
            'args': args,
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
            'acclist_train':acclist_train,
            'acclist_val':acclist_val,
            }, args=args, is_best=is_best)

    if args.export:
        torch.save(model.module.state_dict(), args.export + '.pth')
        return

    if args.eval:
        validate(args.start_epoch)
        return

    for epoch in range(args.start_epoch, args.epochs):
        train(epoch)
        validate(epoch)

    validate(epoch)

if __name__ == "__main__":
    main()
