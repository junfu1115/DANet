##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np

import torch
from torch.autograd import Variable, gradcheck

import encoding

EPS = 1e-3
ATOL = 1e-3

def _assert_tensor_close(a, b, atol=ATOL, rtol=EPS):
    npa, npb = a.cpu().numpy(), b.cpu().numpy()
    assert np.allclose(npa, npb, rtol=rtol, atol=atol), \
        'Tensor close check failed\n{}\n{}\nadiff={}, rdiff={}'.format(
            a, b, np.abs(npa - npb).max(), np.abs((npa - npb) / np.fmax(npa, 1e-5)).max())

def test_aggregate():
    B,N,K,D = 2,3,4,5
    A = Variable(torch.cuda.DoubleTensor(B,N,K).uniform_(-0.5,0.5), 
        requires_grad=True)
    X = Variable(torch.cuda.DoubleTensor(B,N,D).uniform_(-0.5,0.5), 
        requires_grad=True)
    C = Variable(torch.cuda.DoubleTensor(K,D).uniform_(-0.5,0.5), 
        requires_grad=True)
    input = (A, X, C)
    test = gradcheck(encoding.functions.aggregate, input, eps=EPS, atol=ATOL)
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
    test = gradcheck(encoding.functions.scaledL2, input, eps=EPS, atol=ATOL)
    print('Testing scaledL2(): {}'.format(test))


def test_encoding():
    B,C,H,W,K = 2,3,4,5,6
    X = Variable(torch.cuda.DoubleTensor(B,C,H,W).uniform_(-0.5,0.5), 
        requires_grad=True)
    input = (X,)
    layer = encoding.nn.Encoding(C,K).double().cuda()
    test = gradcheck(layer, input, eps=EPS, atol=ATOL)
    print('Testing encoding(): {}'.format(test))


def test_sum_square():
    B,C,H = 2,3,4
    X = Variable(torch.cuda.DoubleTensor(B,C,H).uniform_(-0.5,0.5), 
        requires_grad=True)
    input = (X,)
    test = gradcheck(encoding.functions.sum_square, input, eps=EPS, atol=ATOL)
    print('Testing sum_square(): {}'.format(test))


def test_all_reduce():
    ngpu = torch.cuda.device_count()
    X = [torch.DoubleTensor(2,4,4).uniform_(-0.5,0.5).cuda(i) for i in range(ngpu)]
    for x in X:
        x.requires_grad = True
    Y = encoding.parallel.allreduce(1, *X)
    assert (len(X) == len(Y))
    for i in range(1, ngpu):
        _assert_tensor_close(Y[i].data, Y[0].data)
    input = (1, *X)
    test = gradcheck(encoding.parallel.allreduce, input, eps=EPS, atol=ATOL)
    print('Testing allreduce(): {}'.format(test))


def test_syncbn():
    train_mode=True
    # generate input
    B,C,H,W = 8,3,4,5
    X = Variable(torch.cuda.DoubleTensor(B,C,H,W).uniform_(-0.5,0.5), 
                 requires_grad=True)
    input = (X,)
    # SyncBN using DataParallel
    layer = encoding.nn.BatchNorm2d(C)
    model = torch.nn.DataParallel(layer).double().cuda()
    encoding.parallel.patch_replication_callback(model)
    layer.train(train_mode)
    # grad check
    test = gradcheck(model, input, eps=EPS, atol=ATOL)
    print('Testing BatchNorm2d(): {}'.format(test))


def test_syncbn_func():
    # generate input
    B, C, H = 2, 3, 4
    X = Variable(torch.cuda.DoubleTensor(B,C,H).uniform_(-0.5, 0.5), 
        requires_grad=True)
    gamma = Variable(torch.cuda.DoubleTensor(C).uniform_(-0.5, 0.5), requires_grad=True)
    beta = Variable(torch.cuda.DoubleTensor(C).uniform_(-0.5, 0.5), requires_grad=True)
    mean = Variable(torch.cuda.DoubleTensor(C).uniform_(-0.5, 0.5), requires_grad=True)
    std = Variable(torch.cuda.DoubleTensor(C).uniform_(-0.5, 0.5), requires_grad=True)
    N = B * H
    inputs = (X, mean, std, gamma, beta)
    # grad check
    test = gradcheck(encoding.functions.batchnormtrain, inputs, eps=EPS, atol=ATOL)
    print('Testing batchnorm(): {}'.format(test))


def testSyncBN():
    def _checkBatchNormResult(bn1, bn2, input, is_train, cuda=False):
        def _find_bn(module):
            for m in module.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                  encoding.nn.BatchNorm1d, encoding.nn.BatchNorm2d)):
                    return m
        def _syncParameters(bn1, bn2):
            bn1.reset_parameters()
            bn2.reset_parameters()
            if bn1.affine and bn2.affine:
                bn2.weight.data.copy_(bn1.weight.data)
                bn2.bias.data.copy_(bn1.bias.data)
                bn2.running_mean.copy_(bn1.running_mean)
                bn2.running_var.copy_(bn1.running_var)

        bn1.train(mode=is_train)
        bn2.train(mode=is_train)

        if cuda:
            input = input.cuda()
        # using the same values for gamma and beta
        _syncParameters(_find_bn(bn1), _find_bn(bn2))

        input1 = Variable(input.clone().detach(), requires_grad=True)
        input2 = Variable(input.clone().detach(), requires_grad=True)
        output1 = bn1(input1)
        output2 = bn2(input2)
        # assert forwarding
        _assert_tensor_close(input1.data, input2.data)
        _assert_tensor_close(output1.data, output2.data)
        if not is_train:
            return
        (output1 ** 2).sum().backward()
        (output2 ** 2).sum().backward()
        _assert_tensor_close(input1.grad.data, input2.grad.data)
        _assert_tensor_close(_find_bn(bn1).running_mean, _find_bn(bn2).running_mean)
        _assert_tensor_close(_find_bn(bn1).running_var, _find_bn(bn2).running_var)


    bn = torch.nn.BatchNorm2d(10).cuda().double()
    sync_bn = encoding.nn.BatchNorm2d(10).double()
    sync_bn = torch.nn.DataParallel(sync_bn).cuda()
    encoding.parallel.patch_replication_callback(sync_bn)
    # check with unsync version
    for i in range(10):
        print(i)
        _checkBatchNormResult(bn, sync_bn, torch.rand(16, 10, 16, 16).double(), True, cuda=True)
        _checkBatchNormResult(bn, sync_bn, torch.rand(16, 10, 16, 16).double(), False, cuda=True)

if __name__ == '__main__':
    import nose
    nose.runmodule()
