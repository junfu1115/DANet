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
from torch.nn.modules.module import Module
from ._ext import encoding_lib

class aggregate(Function):
	def forward(self, A, R):
		# A \in(BxNxK) R \in(BxNxKxD) => E \in(BxNxD)
		B, N, K, D = R.size()
		E = A.new(B,K,D)
		# TODO support cpu backend
		print(encoding_lib)
		encoding_lib.Encoding_Float_aggregate_forward(E, A, R)
		return E

	def backward(self, E):
		# TODO FIXME this is test only
		return E


class Aggregate(Module):
	def forward(self, A, R):
		return aggregate()(A, R)
