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
import torch.nn as nn
from torch.autograd import Function
from ._ext import encoding_lib

class aggregate(Function):
	def forward(self, A, R):
		# A \in(BxNxK) R \in(BxNxKxD) => E \in(BxNxD)
		self.save_for_backward(A, R)
		B, N, K, D = R.size()
		E = A.new(B,K,D)
		# TODO support cpu backend
		encoding_lib.Encoding_Float_aggregate_forward(E, A, R)
		return E

	def backward(self, gradE):
		A, R = self.saved_tensors
		gradA = A.new().resize_as_(A)
		gradR = R.new().resize_as_(R)
		encoding_lib.Encoding_Float_aggregate_backward(gradA, gradR, gradE, 
						A, R)
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
