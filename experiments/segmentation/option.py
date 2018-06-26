###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import os
import argparse
import torch

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch \
            Segmentation')
        # model and dataset 
        parser.add_argument('--model', type=str, default='encnet',
                            help='model name (default: encnet)')
        parser.add_argument('--backbone', type=str, default='resnet50',
                            help='backbone name (default: resnet50)')
        parser.add_argument('--dataset', type=str, default='ade20k',
                            help='dataset name (default: pascal12)')
        parser.add_argument('--data-folder', type=str,
                            default=os.path.join(os.environ['HOME'], 'data'),
                            help='training dataset folder (default: \
                            $(HOME)/data)')
        parser.add_argument('--workers', type=int, default=16,
                            metavar='N', help='dataloader threads')
        parser.add_argument('--base-size', type=int, default=608,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=576,
                            help='crop image size')
        # training hyper params
        parser.add_argument('--aux', action='store_true', default= False,
                            help='Auxilary Loss')
        parser.add_argument('--se-loss', action='store_true', default= False,
                            help='Semantic Encoding Loss SE-loss')
        parser.add_argument('--epochs', type=int, default=None, metavar='N',
                            help='number of epochs to train (default: auto)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='start epochs (default:0)')
        parser.add_argument('--batch-size', type=int, default=None,
                            metavar='N', help='input batch size for \
                            training (default: auto)')
        parser.add_argument('--test-batch-size', type=int, default=None,
                            metavar='N', help='input batch size for \
                            testing (default: same as batch size)')
        # optimizer params
        parser.add_argument('--lr', type=float, default=None, metavar='LR',
                            help='learning rate (default: auto)')
        parser.add_argument('--lr-scheduler', type=str, default='poly',
                            help='learning rate scheduler (default: poly)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=1e-4,
                            metavar='M', help='w-decay (default: 1e-4)')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', default=
                            False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default='default',
                            help='set the checkpoint name')
        parser.add_argument('--model-zoo', type=str, default=None,
                            help='evaluating on model zoo model')
        # finetuning pre-trained models
        parser.add_argument('--ft', action='store_true', default= False,
                            help='finetuning on a different dataset')
        parser.add_argument('--pre-class', type=int, default=None,
                            help='num of pre-trained classes \
                            (default: None)')
        # evaluation option
        parser.add_argument('--ema', action='store_true', default= False,
                            help='using EMA evaluation')
        parser.add_argument('--eval', action='store_true', default= False,
                            help='evaluating mIoU')
        parser.add_argument('--no-val', action='store_true', default= False,
                            help='skip validation during training')
        # test option
        parser.add_argument('--test-folder', type=str, default=None,
                            help='path to test image folder')
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        # default settings for epochs, batch_size and lr
        if args.epochs is None:
            epoches = {
                'pascal_voc': 50,
                'pascal_aug': 50,
                'pcontext': 80,
                'ade20k': 160,
            }
            args.epochs = epoches[args.dataset.lower()]
        if args.batch_size is None:
            args.batch_size = 4 * torch.cuda.device_count()
        if args.test_batch_size is None:
            args.test_batch_size = args.batch_size
        if args.lr is None:
            lrs = {
                'pascal_voc': 0.0001,
                'pascal_aug': 0.001,
                'pcontext': 0.001,
                'ade20k': 0.01,
            }
            args.lr = lrs[args.dataset.lower()] / 16 * args.batch_size
        return args
