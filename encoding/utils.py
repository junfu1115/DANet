##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Util Tools"""
import shutil
import os
import math
import torch

__all__ = ['get_optimizer', 'LR_Scheduler', 'save_checkpoint']

def get_optimizer(args, model, diff_LR=True):
    """
    Returns an optimizer for given model,

    Args:
        args: :attr:`args.lr`, :attr:`args.momentum`, :attr:`args.weight_decay`
        model: if using different lr, define `model.pretrained` and `model.head`.
    """
    if diff_LR and model.pretrained is not None:
        print('Using different learning rate for pre-trained features')
        optimizer = torch.optim.SGD([
            {'params': model.pretrained.parameters()},
            {'params': model.head.parameters(),
             'lr': args.lr*10},
            ],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    return optimizer


class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        niters: number of iterations per epoch
    """
    def __init__(self, args, niters=0):
        self.mode = args.lr_scheduler
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = args.lr
        if self.mode == 'step':
            self.lr_step = args.lr_step
        else:
            self.niters = niters
            self.N = args.epochs * niters
        self.epoch = -1

    def __call__(self, optimizer, i, epoch, best_pred):
        if self.mode == 'cos':
            T = (epoch - 1) * self.niters + i
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            T = (epoch - 1) * self.niters + i
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** ((epoch - 1) // self.lr_step))
        else:
            raise RuntimeError('Unknown LR scheduler!')
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.4f, \
                previous best = %.4f' % (epoch, lr, best_pred))
            self.epoch = epoch
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10


# refer to https://github.com/xternalz/WideResNet-pytorch
def save_checkpoint(state, args, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/%s/%s/"%(args.dataset, args.model, args.checkname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth.tar')
