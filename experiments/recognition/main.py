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
import matplotlib.pyplot as plot
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from option import Options
from encoding.utils import *

from tqdm import tqdm

# global variable
best_pred = 100.0
errlist_train = []
errlist_val = []

def main():
    # init the args
    global best_pred, errlist_train, errlist_val
    args = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    # plot 
    if args.plot:
        print('=>Enabling matplotlib for display:')
        plot.ion()
        plot.show()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # init dataloader
    dataset = importlib.import_module('dataset.'+args.dataset)
    Dataloader = dataset.Dataloader
    train_loader, test_loader = Dataloader(args).getloader()
    # init the model
    models = importlib.import_module('model.'+args.model)
    model = models.Net(args)
    print(model)
    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.cuda:
        model.cuda()
        # Please use CUDA_VISIBLE_DEVICES to control the number of gpus
        model = torch.nn.DataParallel(model)
    # check point
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] +1
            best_pred = checkpoint['best_pred']
            errlist_train = checkpoint['errlist_train']
            errlist_val = checkpoint['errlist_val']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            raise RuntimeError ("=> no resume checkpoint found at '{}'".\
                format(args.resume))
    scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs,
                             len(train_loader), args.lr_step)
    def train(epoch):
        model.train()
        global best_pred, errlist_train
        train_loss, correct, total = 0,0,0
        tbar = tqdm(train_loader, desc='\r')
        for batch_idx, (data, target) in enumerate(tbar):
            scheduler(optimizer, batch_idx, epoch, best_pred)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item()
            pred = output.data.max(1)[1] 
            correct += pred.eq(target.data).cpu().sum()
            total += target.size(0)
            err = 100.0 - 100.0 * correct / total
            tbar.set_description('\rLoss: %.3f | Err: %.3f%% (%d/%d)' % \
                (train_loss/(batch_idx+1), err, total-correct, total))

        errlist_train += [err]

    def test(epoch):
        model.eval()
        global best_pred, errlist_train, errlist_val
        test_loss, correct, total = 0,0,0
        is_best = False
        tbar = tqdm(test_loader, desc='\r')
        for batch_idx, (data, target) in enumerate(tbar):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            with torch.no_grad():
                output = model(data)
                test_loss += criterion(output, target).data.item()
                # get the index of the max log-probability
                pred = output.data.max(1)[1] 
                correct += pred.eq(target.data).cpu().sum().item()
                total += target.size(0)

            err = 100.0 - 100.0 * correct / total
            tbar.set_description('Loss: %.3f | Err: %.3f%% (%d/%d)'% \
                (test_loss/(batch_idx+1), err, total-correct, total))

        if args.eval:
            print('Error rate is %.3f'%err)
            return
        # save checkpoint
        errlist_val += [err]
        if err < best_pred:
            best_pred = err 
            is_best = True
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
            'errlist_train':errlist_train,
            'errlist_val':errlist_val,
            }, args=args, is_best=is_best)
        if args.plot:
            plot.clf()
            plot.xlabel('Epoches: ')
            plot.ylabel('Error Rate: %')
            plot.plot(errlist_train, label='train')
            plot.plot(errlist_val, label='val')
            plot.legend(loc='upper left')
            plot.draw()
            plot.pause(0.001)

    if args.eval:
        test(args.start_epoch)
        return

    for epoch in range(args.start_epoch, args.epochs + 1):
        train(epoch)
        test(epoch)

    # save train_val curve to a file
    if args.plot:
        plot.clf()
        plot.xlabel('Epoches: ')
        plot.ylabel('Error Rate: %')
        plot.plot(errlist_train, label='train')
        plot.plot(errlist_val, label='val')
        plot.savefig("runs/%s/%s/"%(args.dataset, args.checkname)
                            +'train_val.jpg')

if __name__ == "__main__":
    main()
