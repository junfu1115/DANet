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


def test_moments():
    B,C,H = 2,3,4
    X = Variable(torch.cuda.DoubleTensor(B,C,H).uniform_(-0.5,0.5), 
        requires_grad=True)
    input = (X,)
    test = gradcheck(encoding.functions.moments, input, eps=EPS, atol=ATOL)
    print('Testing moments(): {}'.format(test))

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
