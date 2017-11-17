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


def test_dilated_densenet():
    net = encoding.dilated.densenet161(True).cuda().eval()
    print(net)
    net2 = models.densenet161(True).cuda().eval()

    x=Variable(torch.Tensor(1,3,224,224).uniform_(-0.5,0.5)).cuda()
    y = net.features(x)
    y2 = net2.features(x)

    print(y[0][0])
    print(y2[0][0])


def test_dilated_avgpool():
    X = Variable(torch.cuda.FloatTensor(1,3,75,75).uniform_(-0.5,0.5))
    input = (X,)
    layer = encoding.nn.DilatedAvgPool2d(kernel_size=2, stride=1, padding=0, dilation=2)
    test = gradcheck(layer, input, eps=1e-6, atol=1e-4)
    print('Testing dilatedavgpool2d(): {}'.format(test))


if __name__ == '__main__':
    test_scaledL2()
    test_encoding() 
    test_aggregate()
    test_sum_square()
    test_dilated_avgpool()
    """
    test_dilated_densenet()
    """
