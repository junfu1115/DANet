##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from ._ext import encoding_lib

class aggregateE(Function):
    def forward(self, A, X, C):
        # A \in(BxNxK) R \in(BxNxKxD) => E \in(BxNxD)
        self.save_for_backward(A, X, C)
        B, N, K = A.size()
        D = X.size(2)
        with torch.cuda.device_of(A):
            E = A.new(B,K,D)
        if isinstance(A, torch.cuda.FloatTensor):
            with torch.cuda.device_of(A):
                encoding_lib.Encoding_Float_aggregateE_forward(E, A, X, C)
        elif isinstance(A, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(A):
                encoding_lib.Encoding_Double_aggregateE_forward(E, A, X, C)
        else:
            raise RuntimeError('Unimplemented data type!')
        return E

    def backward(self, gradE):
        A, X, C = self.saved_tensors
        with torch.cuda.device_of(A):
            gradA = A.new().resize_as_(A)
            gradX = A.new().resize_as_(X)
            gradC = A.new().resize_as_(C)
        if isinstance(A, torch.cuda.FloatTensor):
            with torch.cuda.device_of(A):
                encoding_lib.Encoding_Float_aggregateE_backward(gradA, 
                    gradE, A, X, C)
        elif isinstance(A, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(A):
                encoding_lib.Encoding_Double_aggregateE_backward(gradA, 
                    gradE, A, X, C)
        else:
            raise RuntimeError('Unimplemented data type!')
        gradX.copy_(torch.bmm(A, gradE))
        gradC.copy_((-gradE*A.sum(1).unsqueeze(2)).sum(0))
        return gradA, gradX, gradC


class ScaledL2(Function):
    def forward(self, X, C, S):
        B,N,D = X.size()
        K = C.size(0)
        with torch.cuda.device_of(X):
            SL = X.new(B,N,K)
        if isinstance(X, torch.cuda.FloatTensor):
            with torch.cuda.device_of(X):
                encoding_lib.Encoding_Float_scaledl2_forward(SL, X, C, S)
        elif isinstance(X, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(X):
                encoding_lib.Encoding_Double_scaledl2_forward(SL, X, C, S)
        else:
            raise RuntimeError('Unimplemented data type!')
        self.save_for_backward(X, C, S, SL)
        return SL
    def backward(self, gradSL):
        X, C, S, SL = self.saved_tensors
        K = C.size(0)
        with torch.cuda.device_of(X):
            gradX = X.new().resize_as_(X)
            gradC = X.new().resize_as_(C)
            gradS = X.new().resize_as_(S)
        if isinstance(X, torch.cuda.FloatTensor):
            with torch.cuda.device_of(X):
                encoding_lib.Encoding_Float_scaledl2_backward(gradSL, 
                    gradX, gradC, X, C, S)
        elif isinstance(X, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(X):
                encoding_lib.Encoding_Double_scaledl2_backward(gradSL, 
                    gradX, gradC, X, C, S)
        else:
            raise RuntimeError('Unimplemented data type!')
        gradS.copy_((gradSL*(SL/S.view(1,1,K))).sum(0).sum(0))
        return gradX, gradC, gradS


class Encoding(nn.Module):
    def __init__(self, D, K):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = nn.Parameter(torch.Tensor(K, D), 
            requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True) 
        self.reset_params()
        
    def reset_params(self):
        std1 = 1./((self.K*self.D)**(1/2))
        std2 = 1./((self.K)**(1/2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-std2, std2)

    def forward(self, X):
        # input X is a 4D tensor
        assert(X.size(1)==self.D,"Encoding Layer wrong channels!")
        if X.dim() == 3:
            # BxDxN
            B, N, K, D = X.size(0), X.size(2), self.K, self.D
            X = X.transpose(1,2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW
            B, N, K, D = X.size(0), X.size(2)*X.size(3), self.K, self.D
            X = X.view(B,D,-1).transpose(1,2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        # assignment weights
        A = F.softmax(ScaledL2()(X, self.codewords, self.scale))
        # aggregate
        E = aggregateE()(A, X, self.codewords)
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.D) + '=>' + str(self.K) + 'x' \
            + str(self.D) + ')'


##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class aggregate(Function):
    def forward(self, A, R):
        # A \in(BxNxK) R \in(BxNxKxD) => E \in(BxNxD)
        self.save_for_backward(A, R)
        B, N, K, D = R.size()
        with torch.cuda.device_of(A):
            E = A.new(B,K,D)
        if isinstance(A, torch.cuda.FloatTensor):
            with torch.cuda.device_of(A):
                encoding_lib.Encoding_Float_aggregate_forward(E, A, R)
        elif isinstance(A, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(A):
                encoding_lib.Encoding_Double_aggregate_forward(E, A, R)
        else:
            raise RuntimeError('Unimplemented data type!')
        return E

    def backward(self, gradE):
        A, R = self.saved_tensors
        with torch.cuda.device_of(A):
            gradA = A.new().resize_as_(A)
            gradR = R.new().resize_as_(R)
        if isinstance(A, torch.cuda.FloatTensor):
            with torch.cuda.device_of(A):
                encoding_lib.Encoding_Float_aggregate_backward(gradA, 
                    gradR, gradE, A, R)
        elif isinstance(A, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(A):
                encoding_lib.Encoding_Double_aggregate_backward(gradA, 
                    gradR, gradE, A, R)
        else:
            raise RuntimeError('Unimplemented data type!')
        return gradA, gradR


class residual(Function):
    def forward(self, X, C):
        # X \in(BxNxD) D \in(KxD) R \in(BxNxKxD) 
        B, N, D = X.size()
        K = C.size(0)
        with torch.cuda.device_of(X):
            R = X.new(B,N,K,D)
        if isinstance(X, torch.cuda.FloatTensor):
            with torch.cuda.device_of(X):
                encoding_lib.Encoding_Float_residual_forward(R, X, C)
        elif isinstance(X, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(X):
                encoding_lib.Encoding_Double_residual_forward(R, X, C)
        else:
            raise RuntimeError('Unimplemented data type!')
        return R

    def backward(self, gradR):
        B, N, K, D = gradR.size()
        with torch.cuda.device_of(gradR):
            gradX = gradR.new(B,N,D)
            gradD = gradR.new(K,D)
        if isinstance(gradR, torch.cuda.FloatTensor):
            with torch.cuda.device_of(gradR):
                encoding_lib.Encoding_Float_residual_backward(gradR, 
                    gradX, gradD)
        elif isinstance(gradR, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(gradR):
                encoding_lib.Encoding_Double_residual_backward(gradR, 
                    gradX, gradD)
        else:
            raise RuntimeError('Unimplemented data type!')
        return gradX, gradD


class square_squeeze(Function):
    def forward(self, R):
        B, N, K, D = R.size()
        with torch.cuda.device_of(R):
            L = R.new(B,N,K)
        if isinstance(R, torch.cuda.FloatTensor):
            with torch.cuda.device_of(R):
                encoding_lib.Encoding_Float_squaresqueeze_forward(L, R)
        elif isinstance(R, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(R):
                encoding_lib.Encoding_Double_squaresqueeze_forward(L, R)
        else:
            raise RuntimeError('Unimplemented data type!')
        self.save_for_backward(L, R)
        return L

    def backward(self, gradL):
        L, R = self.saved_tensors
        B, N, K, D = R.size()
        with torch.cuda.device_of(R):
            gradR = R.new(B,N,K,D)
        if isinstance(R, torch.cuda.FloatTensor):
            with torch.cuda.device_of(gradL):
                encoding_lib.Encoding_Float_squaresqueeze_backward(gradL, 
                    gradR, R)
        elif isinstance(R, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(gradL):
                encoding_lib.Encoding_Double_squaresqueeze_backward(gradL, 
                    gradR, R)
        else:
            raise RuntimeError('Unimplemented data type!')
        return gradR
    

def assign(R, S):
    L = square_squeeze()(R)
    K = S.size(0)
    SL = L * S.view(1,1,K)
    return F.softmax(SL)


class Aggregate(nn.Module):
    def forward(self, A, R):
        return aggregate()(A, R)


class EncodingP(nn.Module):
    def __init__(self, D, K):
        super(EncodingP, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = nn.Parameter(torch.Tensor(K, D), 
            requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True) 
        self.reset_params()
        
    def reset_params(self):
        std1 = 1./((self.K*self.D)**(1/2))
        std2 = 1./((self.K)**(1/2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-std2, std2)

    def forward(self, X):
        # input X is a 4D tensor
        assert(X.size(1)==self.D,"Encoding Layer wrong channels!")
        if X.dim() == 3:
            # BxDxN
            B, N, K, D = X.size(0), X.size(2), self.K, self.D
            X = X.transpose(1,2)
        elif X.dim() == 4:
            # BxDxHxW
            B, N, K, D = X.size(0), X.size(2)*X.size(3), self.K, self.D
            X = X.view(B,D,-1).transpose(1,2)
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        # calculate residuals
        R = residual()(X.contiguous(), self.codewords)
        # assignment weights
        A = assign(R, self.scale)
        # aggregate
        E = aggregate()(A, R)

        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.D) + '=>' + str(self.K) + 'x' \
            + str(self.D) + ')'


##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class sum_square(Function):
    def forward(ctx, input):
        ctx.save_for_backward(input)
        B,C,H,W = input.size()
        with torch.cuda.device_of(input):
            xsum    = input.new().resize_(C).zero_()
            xsquare = input.new().resize_(C).zero_()
        if isinstance(input, torch.cuda.FloatTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Float_sum_square_Forward(
                    input.view(B,C,-1), xsum, xsquare)
        elif isinstance(input, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Double_sum_square_Forward( 
                    input.view(B,C,-1), xsum, xsquare)
        else:
            raise RuntimeError('Unimplemented data type!') 
        return xsum, xsquare

    def backward(ctx, gradSum, gradSquare):
        input, = ctx.saved_tensors
        B,C,H,W = input.size()
        with torch.cuda.device_of(input):
            gradInput = input.new().resize_(B,C,H*W).zero_()
        if isinstance(input, torch.cuda.FloatTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Float_sum_square_Backward(
                    gradInput, input.view(B,C,-1), gradSum, gradSquare)
        elif isinstance(input, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Double_sum_square_Backward( 
                    gradInput, input.view(B,C,-1), gradSum, gradSquare)
        else:
            raise RuntimeError('Unimplemented data type!') 
        return gradInput.view(B,C,H,W)


class batchnormtrain(Function):
    def forward(ctx, input, gamma, beta, mean, std):
        ctx.save_for_backward(input, gamma, beta, mean, std)
        assert(input.dim()==3)
        with torch.cuda.device_of(input):
            invstd = 1.0 / std
            output = input.new().resize_as_(input)
        if isinstance(input, torch.cuda.FloatTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Float_batchnorm_Forward(output, 
                    input, mean, invstd, gamma, beta)
        elif isinstance(input, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Double_batchnorm_Forward(output, 
                    input, mean, invstd, gamma, beta)
        else:
            raise RuntimeError('Unimplemented data type!')
        return output 

    def backward(ctx, gradOutput):
        input, gamma, beta, mean, std = ctx.saved_tensors
        invstd = 1.0 / std
        with torch.cuda.device_of(input):
            gradInput = gradOutput.new().resize_as_(input).zero_()
            gradGamma = gradOutput.new().resize_as_(gamma).zero_()
            gradBeta  = gradOutput.new().resize_as_(beta).zero_()
            gradMean  = gradOutput.new().resize_as_(mean).zero_()
            gradStd   = gradOutput.new().resize_as_(std).zero_()

        if isinstance(input, torch.cuda.FloatTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Float_batchnorm_Backward(
                    gradOutput, input, gradInput, gradGamma, gradBeta, 
                    mean, invstd, gamma, beta, gradMean, gradStd,
                    True) 
        elif isinstance(input, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Double_batchnorm_Backward(
                    gradOutput, input, gradInput, gradGamma, gradBeta, 
                    mean, invstd, gamma, beta, gradMean, gradStd,
                    True) 
        else:
            raise RuntimeError('Unimplemented data type!')
        return gradInput, gradGamma, gradBeta, gradMean, gradStd


class batchnormeval(Function):
    def forward(ctx, input, gamma, beta, mean, std):
        ctx.save_for_backward(input, gamma, beta, mean, std)
        assert(input.dim()==3)
        with torch.cuda.device_of(input):
            invstd = 1.0 / std
            output = input.new().resize_as_(input)
        if isinstance(input, torch.cuda.FloatTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Float_batchnorm_Forward(output, 
                    input, mean, invstd, gamma, beta)
        elif isinstance(input, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Double_batchnorm_Forward(output, 
                    input, mean, invstd, gamma, beta)
        else:
            raise RuntimeError('Unimplemented data type!')
        return output 

    def backward(ctx, gradOutput):
        input, gamma, beta, mean, std = ctx.saved_tensors
        invstd = 1.0 / std
        with torch.cuda.device_of(input):
            gradInput = gradOutput.new().resize_as_(input).zero_()
            gradGamma = gradOutput.new().resize_as_(gamma).zero_()
            gradBeta  = gradOutput.new().resize_as_(beta).zero_()
            gradMean  = gradOutput.new().resize_as_(mean).zero_()
            gradStd   = gradOutput.new().resize_as_(std).zero_()
        if isinstance(input, torch.cuda.FloatTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Float_batchnorm_Backward(
                    gradOutput, input, gradInput, gradGamma, gradBeta, 
                    mean, invstd, gamma, beta, gradMean, gradStd,
                    False) 
        elif isinstance(input, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Double_batchnorm_Backward(
                    gradOutput, input, gradInput, gradGamma, gradBeta, 
                    mean, invstd, gamma, beta, gradMean, gradStd,
                    False) 
        else:
            raise RuntimeError('Unimplemented data type!')
        return gradInput, gradGamma, gradBeta, gradMean, gradStd

