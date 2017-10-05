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

def test_aggregate():
    B,N,K,D = 2,3,4,5
    A = Variable(torch.cuda.DoubleTensor(B,N,K).uniform_(-0.5,0.5), 
        requires_grad=True)
    R = Variable(torch.cuda.DoubleTensor(B,N,K,D).uniform_(-0.5,0.5), 
        requires_grad=True)
    input = (A, R)
    test = gradcheck(encoding.aggregate(), input, eps=1e-6, atol=1e-4)
    print('Testing aggregate(): {}'.format(test))


def test_aggregateE():
    B,N,K,D = 2,3,4,5
    A = Variable(torch.cuda.DoubleTensor(B,N,K).uniform_(-0.5,0.5), 
        requires_grad=True)
    X = Variable(torch.cuda.DoubleTensor(B,N,D).uniform_(-0.5,0.5), 
        requires_grad=True)
    C = Variable(torch.cuda.DoubleTensor(K,D).uniform_(-0.5,0.5), 
        requires_grad=True)
    input = (A, X, C)
    test = gradcheck(encoding.aggregateE(), input, eps=1e-6, atol=1e-4)
    print('Testing aggregateE(): {}'.format(test))


def test_ScaledL2():
    B,N,K,D = 2,3,4,5
    X = Variable(torch.cuda.DoubleTensor(B,N,D).uniform_(-0.5,0.5), 
        requires_grad=True)
    C = Variable(torch.cuda.DoubleTensor(K,D).uniform_(-0.5,0.5), 
        requires_grad=True)
    S = Variable(torch.cuda.DoubleTensor(K).uniform_(-0.5,0.5), 
        requires_grad=True)
    input = (X, C, S)
    test = gradcheck(encoding.ScaledL2(), input, eps=1e-6, atol=1e-4)
    print('Testing ScaledL2(): {}'.format(test))


def test_assign():
    B,N,K,D = 2,3,4,5
    X = Variable(torch.cuda.DoubleTensor(B,N,D).uniform_(-0.5,0.5), 
        requires_grad=True)
    C = Variable(torch.cuda.DoubleTensor(K,D).uniform_(-0.5,0.5), 
        requires_grad=True)
    S = Variable(torch.cuda.DoubleTensor(K).uniform_(-0.5,0.5), 
        requires_grad=True)

    R = encoding.residual()(X, C)
    A1 = encoding.assign(R, S)
    E1 = encoding.aggregate()(A1, R)

    A2 = F.softmax(encoding.ScaledL2()(X,C,S))
    E2 = encoding.aggregateE()(A2, X, C)

    print('E1', E1)
    print('E2', E2)


def test_residual():
    B,N,K,D = 2,3,4,5
    X = Variable(torch.cuda.DoubleTensor(B,N,D).uniform_(-0.5,0.5), 
        requires_grad=True)
    C = Variable(torch.cuda.DoubleTensor(K,D).uniform_(-0.5,0.5), 
        requires_grad=True)
    input = (X, C)
    test = gradcheck(encoding.residual(), input, eps=1e-6, atol=1e-4)
    print('Testing residual(): {}'.format(test))


def test_square_squeeze():
    B,N,K,D = 2,3,4,5
    R = Variable(torch.cuda.DoubleTensor(B,N,K,D).uniform_(-0.5,0.5), 
        requires_grad=True)
    input = (R,)
    test = gradcheck(encoding.square_squeeze(), input, eps=1e-6, atol=1e-4)
    print('Testing square_squeeze(): {}'.format(test))


def test_encoding():
    B,C,H,W,K = 2,3,4,5,6
    X = Variable(torch.cuda.DoubleTensor(B,C,H,W).uniform_(-0.5,0.5), 
        requires_grad=True)
    input = (X,)
    layer = encoding.Encoding(C,K).double().cuda()
    test = gradcheck(layer, input, eps=1e-6, atol=1e-4)
    print('Testing encoding(): {}'.format(test))
    

def test_encodingP():
    B,C,H,W,K = 2,3,4,5,6
    X = Variable(torch.cuda.DoubleTensor(B,C,H,W).uniform_(-0.5,0.5), 
        requires_grad=True)
    input = (X,)
    layer = encoding.EncodingP(C,K).double().cuda()
    test = gradcheck(layer, input, eps=1e-6, atol=1e-4)
    print('Testing encodingP(): {}'.format(test))


def test_sum_square():
    B,C,H,W = 2,3,4,5
    X = Variable(torch.cuda.DoubleTensor(B,C,H,W).uniform_(-0.5,0.5), 
        requires_grad=True)
    input = (X,)
    test = gradcheck(encoding.sum_square(), input, eps=1e-6, atol=1e-4)
    print('Testing sum_square(): {}'.format(test))


if __name__ == '__main__':
    test_aggregateE()
    test_ScaledL2()
    test_encoding() 
    test_aggregate()
    test_residual()
    #test_assign()
    test_square_squeeze()
    test_encodingP()
    test_sum_square()
