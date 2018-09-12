##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse
import os

class Options():
    def __init__(self):
        # Training settings
        parser = argparse.ArgumentParser(description='Deep Encoding')
        parser.add_argument('--dataset', type=str, default='cifar10',
            help='training dataset (default: cifar10)')
        # model params 
        parser.add_argument('--model', type=str, default='densenet',
            help='network model type (default: densenet)')
        parser.add_argument('--nclass', type=int, default=10, metavar='N',
            help='number of classes (default: 10)')
        parser.add_argument('--widen', type=int, default=4, metavar='N',
            help='widen factor of the network (default: 4)')
        parser.add_argument('--ncodes', type=int, default=32, metavar='N',
            help='number of codewords in Encoding Layer (default: 32)')
        parser.add_argument('--backbone', type=str, default='resnet50',
            help='backbone name (default: resnet50)')
        # training hyper params
        parser.add_argument('--batch-size', type=int, default=128,
            metavar='N', help='batch size for training (default: 128)')
        parser.add_argument('--test-batch-size', type=int, default=256, 
            metavar='N', help='batch size for testing (default: 256)')
        parser.add_argument('--epochs', type=int, default=600, metavar='N',
            help='number of epochs to train (default: 600)')
        parser.add_argument('--start_epoch', type=int, default=1, 
            metavar='N', help='the epoch number to start (default: 0)')
        # lr setting
        parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
            help='learning rate (default: 0.1)')
        parser.add_argument('--lr-scheduler', type=str, default='step', 
            help='learning rate scheduler (default: step)')
        parser.add_argument('--lr-step', type=int, default=40, metavar='LR',
            help='learning rate step (default: 40)')
        # optimizer
        parser.add_argument('--momentum', type=float, default=0.9, 
            metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=5e-4, 
            metavar ='M', help='SGD weight decay (default: 5e-4)')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', 
            default=False, help='disables CUDA training')
        parser.add_argument('--plot', action='store_true', default=False,
            help='matplotlib')
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
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        if args.dataset == 'minc':
            args.nclass = 23
        return args
