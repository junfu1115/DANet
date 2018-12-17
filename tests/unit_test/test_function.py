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

def test_scaled_l2():
    B,N,K,D = 2,3,4,5
    X = Variable(torch.cuda.DoubleTensor(B,N,D).uniform_(-0.5,0.5), 
        requires_grad=True)
    C = Variable(torch.cuda.DoubleTensor(K,D).uniform_(-0.5,0.5), 
        requires_grad=True)
    S = Variable(torch.cuda.DoubleTensor(K).uniform_(-0.5,0.5), 
        requires_grad=True)
    input = (X, C, S)
    test = gradcheck(encoding.functions.scaled_l2, input, eps=EPS, atol=ATOL)
    print('Testing scaled_l2(): {}'.format(test))

def test_aggregate_v2():
    def py_aggregate_v2(A, X, C, STD, S):
        B, N, D = X.size()
        K = C.size(0)
        #e_{k} = \sum_{i=1}^{N} a_{ik} (x_i - d_k) / \sigma_k
        R = (X.view(B, N, 1, D).expand(B, N, K, D) - \
             C.view(1, 1, K, D).expand(B, N, K, D)) / STD.view(1, 1, K, D)
        #E = 1.0 / torch.sqrt(S + 1e-5).unsqueeze(0).unsqueeze(2) * (A.unsqueeze(3) * R).sum(1)
        E2 = (A.unsqueeze(3) * R).sum(1)
        return E2

    B,N,K,D = 2,3,4,5
    A = Variable(torch.cuda.DoubleTensor(B,N,K).uniform_(-0.5,0.5), 
                 requires_grad=True)
    X = Variable(torch.cuda.DoubleTensor(B,N,D).uniform_(-0.5,0.5), 
                 requires_grad=True)
    C = Variable(torch.cuda.DoubleTensor(K,D).uniform_(-0.5,0.5), 
                 requires_grad=True)
    STD = Variable(torch.cuda.DoubleTensor(K,D).uniform_(-0.5,0.5), 
                   requires_grad=True)
    S = Variable(torch.cuda.DoubleTensor(K).uniform_(-0.5,0.5), 
                 requires_grad=True)

    A2 = torch.from_numpy(A.detach().cpu().numpy()).cuda()
    X2 = torch.from_numpy(X.detach().cpu().numpy()).cuda()
    C2 = torch.from_numpy(C.detach().cpu().numpy()).cuda()
    STD2 = torch.from_numpy(STD.detach().cpu().numpy()).cuda()
    S2 = torch.from_numpy(S.detach().cpu().numpy()).cuda()
    A2.requires_grad_()
    X2.requires_grad_()
    C2.requires_grad_()
    STD2.requires_grad_()
    S2.requires_grad_()

    E = encoding.functions.aggregate_v2(A, X, C, STD)
    E2 = py_aggregate_v2(A2, X2, C2, STD2, S2)
    _assert_tensor_close(E.detach(), E2.detach())

    input = (A, X, C, STD)
    test = gradcheck(encoding.functions.aggregate_v2, input, eps=EPS, atol=ATOL)
    print('Testing aggregate_v2(): {}'.format(test))

def test_encoding_dist():
    def mahalanobis_dist(X, C):
        B, N, D = X.size()
        K = C.size(0)
        # X \in BxNxD, C \in KxD
        R = X.view(B, N, 1, D).expand(B, N, K, D) - \
            C.view(1, 1, K, D).expand(B, N, K, D)
        STD = torch.sqrt(R.pow(2).mean(0).mean(0) + 1e-6)
        KD = (R / STD.view(1,1,K,D)).pow(2).sum(3)
        return KD, STD

    B,N,K,D = 2,3,4,5
    RVar = torch.cuda.DoubleTensor(K,D).zero_()
    X = torch.cuda.DoubleTensor(B,N,D).uniform_(-0.5,0.5)
    C = torch.cuda.DoubleTensor(K,D).uniform_(-0.5,0.5)
    X.requires_grad_()
    C.requires_grad_()

    X2 = torch.from_numpy(X.detach().cpu().numpy()).cuda()
    C2 = torch.from_numpy(C.detach().cpu().numpy()).cuda()
    X2.requires_grad_()
    C2.requires_grad_()
    # assert numeric correctness
    KD, STD, Var_ = encoding.functions.encoding_dist(X, C, 1e-6)
    KD2, STD2 = mahalanobis_dist(X2, C2)
    _assert_tensor_close(STD.detach(), STD2.detach())
    _assert_tensor_close(KD.detach(), KD2.detach())
    # check backward
    loss1 = KD.pow(2).sum() + STD.sum()
    loss1.backward()
    loss2 = KD2.pow(2).sum() + STD2.sum()
    loss2.backward()
    _assert_tensor_close(X.grad.detach(), X2.grad.detach())
    _assert_tensor_close(C.grad.detach(), C2.grad.detach())

    input = (X, C, 1e-6)
    test = gradcheck(encoding.functions.encoding_dist, input, eps=EPS, atol=ATOL)
    print('Testing encoding_dist(): {}'.format(test))

def test_encoding_dist_inference():
    def mahalanobis_dist(X, C, STD):
        B, N, D = X.size()
        K = C.size(0)
        # X \in BxNxD, C \in KxD
        R = X.view(B, N, 1, D).expand(B, N, K, D) - \
            C.view(1, 1, K, D).expand(B, N, K, D)
        #STD = torch.sqrt(R.pow(2).mean(0).mean(0) + 1e-6)
        KD = (R / STD.view(1,1,K,D)).pow(2).sum(3)
        return KD

    B,N,K,D = 2,3,4,5
    X = Variable(torch.cuda.DoubleTensor(B,N,D).uniform_(-0.5,0.5), 
                 requires_grad=True)
    C = Variable(torch.cuda.DoubleTensor(K,D).uniform_(-0.5,0.5), 
                 requires_grad=True)
    STD = Variable(torch.cuda.DoubleTensor(K,D).uniform_(-0.5,0.5), 
                   requires_grad=True)
    
    X2 = torch.from_numpy(X.detach().cpu().numpy()).cuda()
    C2 = torch.from_numpy(C.detach().cpu().numpy()).cuda()
    STD2 = torch.from_numpy(STD.detach().cpu().numpy()).cuda()
    X2.requires_grad_()
    C2.requires_grad_()
    STD2.requires_grad_()

    E = encoding.functions.encoding_dist_inference(X, C, STD)
    E2 = mahalanobis_dist(X2, C2, STD2)

    loss1 = E.pow(2).sum()
    loss2 = E2.pow(2).sum()
    loss1.backward()
    loss2.backward()

    print('X.grad', X.grad)
    print('X2.grad', X2.grad)

    _assert_tensor_close(E.detach(), E2.detach())
    _assert_tensor_close(X.grad.detach(), X2.grad.detach())
    _assert_tensor_close(C.grad.detach(), C2.grad.detach())
    _assert_tensor_close(STD.grad.detach(), STD2.grad.detach())

    input = (X, C, STD)
    test = gradcheck(encoding.functions.encoding_dist_inference, input, eps=EPS, atol=ATOL)
    print('Testing encoding_dist_inference(): {}'.format(test))

def test_moments():
    B,C,H = 2,3,4
    X = Variable(torch.cuda.DoubleTensor(B,C,H).uniform_(-0.5,0.5), 
        requires_grad=True)
    input = (X,)
    test = gradcheck(encoding.functions.moments, input, eps=EPS, atol=ATOL)
    print('Testing moments(): {}'.format(test))

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

def test_non_max_suppression():
    def _test_nms(cuda):
        # check a small test case
        boxes = torch.Tensor([
            [[10.2, 23., 50., 20.],
             [11.3, 23., 52., 20.1],
             [23.2, 102.3, 23.3, 50.3],
             [101.2, 32.4, 70.6, 70.],
             [100.2, 30.9, 70.7, 69.]],
            [[200.3, 234., 530., 320.],
             [110.3, 223., 152., 420.1],
             [243.2, 240.3, 50.3, 30.3],
             [243.2, 236.4, 48.6, 30.],
             [100.2, 310.9, 170.7, 691.]]])

        scores = torch.Tensor([
            [0.9, 0.7, 0.11, 0.23, 0.8],
            [0.13, 0.89, 0.45, 0.23, 0.3]])

        if cuda:
            boxes = boxes.cuda()
            scores = scores.cuda()

        expected_output = (
            torch.ByteTensor(
                [[1, 1, 0, 0, 1], [1, 1, 1, 0, 1]]),
            torch.LongTensor(
                [[0, 4, 1, 3, 2], [1, 2, 4, 3, 0]])
        )

        mask, inds = encoding.functions.NonMaxSuppression(boxes, scores, 0.7)
        _assert_tensor_close(mask, expected_output[0])
        _assert_tensor_close(inds, expected_output[1])

    _test_nms(False)
    _test_nms(True)

if __name__ == '__main__':
    import nose
    nose.runmodule()
