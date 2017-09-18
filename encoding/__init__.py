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
import torch.cuda.nccl as nccl
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
from ._ext import encoding_lib

class aggregate(Function):
	def forward(self, A, R):
		# A \in(BxNxK) R \in(BxNxKxD) => E \in(BxNxD)
		self.save_for_backward(A, R)
		B, N, K, D = R.size()
		E = A.new(B,K,D)
		# TODO support cpu backend
    if isinstance(A, torch.cuda.FloatTensor):
		    encoding_lib.Encoding_Float_aggregate_forward(E, A, R)
    elif isinstance(A, torch.cuda.DoubleTensor):
		    encoding_lib.Encoding_Double_aggregate_forward(E, A, R)
    else:
        raise RuntimeError('unimplemented')
		return E

	def backward(self, gradE):
		A, R = self.saved_tensors
		gradA = A.new().resize_as_(A)
		gradR = R.new().resize_as_(R)
    if isinstance(A, torch.cuda.FloatTensor):
        encoding_lib.Encoding_Float_aggregate_backward(gradA, gradR, gradE, 
                A, R)
    elif isinstance(A, torch.cuda.DoubleTensor):
        encoding_lib.Encoding_Double_aggregate_backward(gradA, gradR, gradE, 
                A, R)
    else:
        raise RuntimeError('unimplemented')
		return gradA, gradR


class Aggregate(nn.Module):
	def forward(self, A, R):
		return aggregate()(A, R)


class Encoding(nn.Module):
	def __init__(self, D, K):
		super(Encoding, self).__init__()
		# init codewords and smoothing factor
		self.D, self.K = D, K
		self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
		self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True) 
		self.softmax = nn.Softmax()
		self.reset_params()
		
	def reset_params(self):
		std1 = 1./((self.K*self.D)**(1/2))
		std2 = 1./((self.K)**(1/2))
		self.codewords.data.uniform_(-std1, std1)
		self.scale.data.uniform_(-std2, std2)

	def forward(self, X):
		# input X is a 4D tensor
		assert(X.size(1)==self.D,"Encoding Layer incompatible input channels!")
		unpacked = False
		if X.dim() == 3:
			unpacked = True
			X = X.unsqueeze(0)

		B, N, K, D = X.size(0), X.size(2)*X.size(3), self.K, self.D
		# reshape input
		X = X.view(B,D,-1).transpose(1,2)
		# calculate residuals
		R = X.contiguous().view(B,N,1,D).expand(B,N,K,D) - self.codewords.view(
					1,1,K,D).expand(B,N,K,D)
		# assignment weights
		A = R
		A = A.pow(2).sum(3).view(B,N,K)
		A = A*self.scale.view(1,1,K).expand_as(A)
		A = self.softmax(A.view(B*N,K)).view(B,N,K)
		# aggregate
		E = aggregate()(A, R)

		if unpacked:
			E = E.squeeze(0)
		return E

	def __repr__(self):
		return self.__class__.__name__ + '(' \
			+ 'N x ' + str(self.D) + '=>' + str(self.K) + 'x' + str(self.D) + ')'

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
            raise RuntimeError('unimplemented') 
        return xsum, xsquare

    def backward(ctx, gradSum, gradSquare):
        input, = ctx.saved_tensors
        B,C,H,W = input.size()
        with torch.cuda.device_of(input):
            gradInput = input.new().resize_(B,C,H*W).zero_()
        #    gradSum.view(1,C,1,1).expand_as(input) + \
        #   2*gradSquare.view(1,C,1,1).expand_as(input)*input
        if isinstance(input, torch.cuda.FloatTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Float_sum_square_Backward(
                    gradInput, input.view(B,C,-1), gradSum, gradSquare)
        elif isinstance(input, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(input):
                encoding_lib.Encoding_Double_sum_square_Backward( 
                    gradInput, input.view(B,C,-1), gradSum, gradSquare)
        else:
            raise RuntimeError('unimplemented') 
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
            raise RuntimeError('unimplemented')
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
            raise RuntimeError('unimplemented')
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
            raise RuntimeError('unimplemented')
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
            raise RuntimeError('unimplemented')
        return gradInput, gradGamma, gradBeta, gradMean, gradStd

