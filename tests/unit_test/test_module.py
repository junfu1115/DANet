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

def test_encoding():
    B,C,H,W,K = 2,3,4,5,6
    X = Variable(torch.cuda.DoubleTensor(B,C,H,W).uniform_(-0.5,0.5), 
        requires_grad=True)
    input = (X,)
    layer = encoding.nn.Encoding(C,K).double().cuda()
    test = gradcheck(layer, input, eps=EPS, atol=ATOL)
    print('Testing encoding(): {}'.format(test))

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

def testSyncBN():
    def _check_batchnorm_result(bn1, bn2, input, is_train, cuda=False):
        def _find_bn(module):
            for m in module.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                  encoding.nn.SyncBatchNorm)):
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
        if is_train:
            bn1.train()
            bn2.train()
            output1 = bn1(input1)
            output2 = bn2(input2)
        else:
            bn1.eval()
            bn2.eval()
            with torch.no_grad():
                output1 = bn1(input1)
                output2 = bn2(input2)
        # assert forwarding
        #_assert_tensor_close(input1.data, input2.data)
        _assert_tensor_close(output1.data, output2.data)
        if not is_train:
            return
        (output1 ** 2).sum().backward()
        (output2 ** 2).sum().backward()
        _assert_tensor_close(_find_bn(bn1).bias.grad.data, _find_bn(bn2).bias.grad.data)
        _assert_tensor_close(_find_bn(bn1).weight.grad.data, _find_bn(bn2).weight.grad.data)
        _assert_tensor_close(input1.grad.data, input2.grad.data)
        _assert_tensor_close(_find_bn(bn1).running_mean, _find_bn(bn2).running_mean)
        #_assert_tensor_close(_find_bn(bn1).running_var, _find_bn(bn2).running_var)

    bn = torch.nn.BatchNorm2d(10).cuda().double()
    sync_bn = encoding.nn.SyncBatchNorm(10, inplace=True, sync=True).cuda().double()
    sync_bn = torch.nn.DataParallel(sync_bn).cuda()
    # check with unsync version
    #_check_batchnorm_result(bn, sync_bn, torch.rand(2, 1, 2, 2).double(), True, cuda=True)
    for i in range(10):
        print(i)
        _check_batchnorm_result(bn, sync_bn, torch.rand(16, 10, 16, 16).double(), True, cuda=True)
        #_check_batchnorm_result(bn, sync_bn, torch.rand(16, 10, 16, 16).double(), False, cuda=True)


def testABN():
    class NormAct(torch.nn.BatchNorm2d):
        def __init__(self, num_features, eps=1e-5, momentum=0.1, sync=True, activation="none",
                     slope=0.01):
            super(NormAct, self).__init__(num_features, eps=eps, momentum=momentum, affine=True)
            self.slope = slope

        def forward(self, x):
            exponential_average_factor = 0.0
            if self.training and self.track_running_stats:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / self.num_batches_tracked.item()
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

            y = torch.nn.functional.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
            return torch.nn.functional.leaky_relu_(y, self.slope)
     
    def _check_batchnorm_result(bn1, bn2, input, is_train, cuda=False):
        def _find_bn(module):
            for m in module.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                  encoding.nn.SyncBatchNorm)):
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
        if is_train:
            bn1.train()
            bn2.train()
            output1 = bn1(input1)
            output2 = bn2(input2)
        else:
            bn1.eval()
            bn2.eval()
            with torch.no_grad():
                output1 = bn1(input1)
                output2 = bn2(input2)
        # assert forwarding
        _assert_tensor_close(output1.data, output2.data)
        if not is_train:
            return
        loss1 = (output1 ** 2).sum()
        loss2 = (output2 ** 2).sum()
        loss1.backward()
        loss2.backward()
        _assert_tensor_close(_find_bn(bn1).bias.grad.data, _find_bn(bn2).bias.grad.data)
        _assert_tensor_close(_find_bn(bn1).weight.grad.data, _find_bn(bn2).weight.grad.data)
        _assert_tensor_close(input1.grad.data, input2.grad.data)
        _assert_tensor_close(_find_bn(bn1).running_mean, _find_bn(bn2).running_mean)

    bn = NormAct(10).cuda().double()
    inp_abn = encoding.nn.SyncBatchNorm(10, sync=False, activation='leaky_relu', inplace=True).cuda().double()
    inp_abn = torch.nn.DataParallel(inp_abn).cuda()
    # check with unsync version
    for i in range(10):
        print(i)
        _check_batchnorm_result(bn, inp_abn, torch.rand(16, 10, 16, 16).double(), True, cuda=True)
        #_check_batchnorm_result(bn, inp_abn, torch.rand(16, 10, 16, 16).double(), False, cuda=True)


def test_Atten_Module():
    B, C, H, W = 8, 24, 10, 10
    X = Variable(torch.cuda.DoubleTensor(B,C,H,W).uniform_(-0.5,0.5), 
                 requires_grad=True)
    layer1 = encoding.nn.MultiHeadAttention(4, 24, 24, 24).double().cuda()
    Y = layer1(X)

if __name__ == '__main__':
    import nose
    nose.runmodule()
