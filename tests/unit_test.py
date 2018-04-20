##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import encoding
import unittest

import torch
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck

import torchvision.models as models

EPS = 1e-6

def test_aggregate():
    B,N,K,D = 2,3,4,5
    A = Variable(torch.cuda.DoubleTensor(B,N,K).uniform_(-0.5,0.5), 
        requires_grad=True)
    X = Variable(torch.cuda.DoubleTensor(B,N,D).uniform_(-0.5,0.5), 
        requires_grad=True)
    C = Variable(torch.cuda.DoubleTensor(K,D).uniform_(-0.5,0.5), 
        requires_grad=True)
    input = (A, X, C)
    test = gradcheck(encoding.functions.aggregate, input, eps=1e-6, atol=1e-4)
    print('Testing aggregate(): {}'.format(test))


def test_scaledL2():
    B,N,K,D = 2,3,4,5
    X = Variable(torch.cuda.DoubleTensor(B,N,D).uniform_(-0.5,0.5), 
        requires_grad=True)
    C = Variable(torch.cuda.DoubleTensor(K,D).uniform_(-0.5,0.5), 
        requires_grad=True)
    S = Variable(torch.cuda.DoubleTensor(K).uniform_(-0.5,0.5), 
        requires_grad=True)
    input = (X, C, S)
    test = gradcheck(encoding.functions.scaledL2, input, eps=1e-6, atol=1e-4)
    print('Testing scaledL2(): {}'.format(test))


def test_encoding():
    B,C,H,W,K = 2,3,4,5,6
    X = Variable(torch.cuda.DoubleTensor(B,C,H,W).uniform_(-0.5,0.5), 
        requires_grad=True)
    input = (X,)
    layer = encoding.nn.Encoding(C,K).double().cuda()
    test = gradcheck(layer, input, eps=1e-6, atol=1e-4)
    print('Testing encoding(): {}'.format(test))


def test_sum_square():
    B,C,H,W = 2,3,4,5
    X = Variable(torch.cuda.DoubleTensor(B,C,H,W).uniform_(-0.5,0.5), 
        requires_grad=True)
    input = (X,)
    test = gradcheck(encoding.functions.sum_square, input, eps=1e-6, atol=1e-4)
    print('Testing sum_square(): {}'.format(test))


def test_all_reduce():
    ngpu = torch.cuda.device_count()
    X = [torch.DoubleTensor(2,4,4).uniform_(-0.5,0.5).cuda(i) for i in range(ngpu)]
    for x in X:
        x.requires_grad = True
    Y = encoding.parallel.allreduce(1, *X)
    assert (len(X) == len(Y))


if __name__ == '__main__':
    import nose
    nose.runmodule()
