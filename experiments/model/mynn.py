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
from torch.autograd import Variable

class Basicblock(nn.Module):
	def __init__(self, inplanes, planes, stride=1, 
							norm_layer=nn.BatchNorm2d):
		super(Basicblock, self).__init__()
		if inplanes != planes*self.expansion or stride !=1 :
			self.downsample = True
			self.residual_layer = nn.Conv2d(inplanes, planes,
														kernel_size=1, stride=stride)
		else:
			self.downsample = False
		conv_block=[]
		conv_block+=[norm_layer(inplanes),
								nn.ReLU(inplace=True),
								nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, 
									padding=1),
								norm_layer(planes),
								nn.ReLU(inplace=True),
								nn.Conv2d(planes, planes, kernel_size=3, stride=1,
									padding=1),
								norm_layer(planes)]
		self.conv_block = nn.Sequential(*conv_block)
	
	def forward(self, input):
		if self.downsample:
			residual = self.residual_layer(input)
		else:
			residual = input
		return residual + self.conv_block(input)


class Bottleneck(nn.Module):
	""" Pre-activation residual block
	Identity Mapping in Deep Residual Networks
	ref https://arxiv.org/abs/1603.05027
	"""
	def __init__(self, inplanes, planes, stride=1,norm_layer=nn.BatchNorm2d):
		super(Bottleneck, self).__init__()
		self.expansion = 4
		if inplanes != planes*self.expansion or stride !=1 :
			self.downsample = True
			self.residual_layer = nn.Conv2d(inplanes, planes * self.expansion,
														kernel_size=1, stride=stride)
		else:
			self.downsample = False
		conv_block = []
		conv_block += [norm_layer(inplanes),
									nn.ReLU(inplace=True),
									nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)]
		conv_block += [norm_layer(planes),
									nn.ReLU(inplace=True),
									nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
										padding=1)]
		conv_block += [norm_layer(planes),
									nn.ReLU(inplace=True),
									nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
										stride=1)]
		self.conv_block = nn.Sequential(*conv_block)
		
	def forward(self, x):
		if self.downsample:
			residual = self.residual_layer(x)
		else:
			residual = x
		return residual + self.conv_block(x)
		
class View(nn.Module):
	def __init__(self, *args):
		super(View, self).__init__()
		if len(args) == 1 and isinstance(args[0], torch.Size):
			self.size = args[0]
		else:
			self.size = torch.Size(args)

	def forward(self, input):
		return input.view(self.size)


class InstanceNormalization(nn.Module):
	"""InstanceNormalization
	Improves convergence of neural-style.
	ref: https://arxiv.org/pdf/1607.08022.pdf
	"""

	def __init__(self, dim, eps=1e-5):
		super(InstanceNormalization, self).__init__()
		self.weight = nn.Parameter(torch.FloatTensor(dim))
		self.bias = nn.Parameter(torch.FloatTensor(dim))
		self.eps = eps
		self._reset_parameters()

	def _reset_parameters(self):
		self.weight.data.uniform_()
		self.bias.data.zero_()

	def forward(self, x):
		n = x.size(2) * x.size(3)
		t = x.view(x.size(0), x.size(1), n)
		mean = torch.mean(t, 2).unsqueeze(2).expand_as(x)
		# Calculate the biased var. torch.var returns unbiased var
		var = torch.var(t, 2).unsqueeze(2).expand_as(x) * ((n - 1) / float(n))
		scale_broadcast = self.weight.unsqueeze(1).unsqueeze(1).unsqueeze(0)
		scale_broadcast = scale_broadcast.expand_as(x)
		shift_broadcast = self.bias.unsqueeze(1).unsqueeze(1).unsqueeze(0)
		shift_broadcast = shift_broadcast.expand_as(x)
		out = (x - mean) / torch.sqrt(var + self.eps)
		out = out * scale_broadcast + shift_broadcast
		return out
