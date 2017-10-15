##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import shutil
import os
import sys
import time
import math

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


class CosLR_Scheduler(object):
    """Cosine Learning Rate Scheduler

    .. math::
        lr = base_lr * 0.5 * (1 + cos(T/N))

    where ``T`` is current iters and ``N`` is total iters

    Args:
        args:  base learning rate :attr:`args.lr`, number of epochs :attr:`args.epochs`
        niters: number of iterations per epoch
    """
    def __init__(self, args, niters):
        self.lr = args.lr
        self.niters = niters
        self.N = args.epochs * niters
        self.epoch = -1

    def __call__(self, optimizer, i, epoch, best_pred):
        T = (epoch - 1) * self.niters + i
        lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        if epoch > self.epoch:
            print('=>Epochs %i, learning rate = %.4f, previous best ='\
                '%.3f%%' % (epoch, lr, best_pred))
            self.epoch = epoch
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        elif len(optimizer.param_groups) == 2:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * 10
        else:
            raise RuntimeError('unsupported number of param groups: {}' \
                .format(len(optimizer.param_groups)))


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

# refer to https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 86.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    """Progress Bar for display
    """
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()    # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('    Step: %s' % _format_time(step_time))
    L.append(' | Tot: %s' % _format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def _format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
