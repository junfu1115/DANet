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
import model.mynn as nn2
from encoding import Encoding

class Net(nn.Module):
	def __init__(self, num_blocks=[2,2,2,2], num_classes=10, 
								block=nn2.Bottleneck):
		super(Net, self).__init__()
		if block == nn2.Basicblock:
			self.expansion = 1
		else:
			self.expansion = 4

		self.inplanes = 64
		num_planes = [64, 128, 256, 512]
		strides = [1, 2, 2, 2]
		model = []
		# Conv_1
		model += [nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1),
							nn.BatchNorm2d(self.inplanes),
							nn.ReLU(inplace=True)]
		# Residual units
		for i in range(4):
			model += [self._residual_unit(block, num_planes[i], num_blocks[i],
								strides[i])]
		# Last conv layer
		# TODO norm layer, instance norm?
		model += [nn.BatchNorm2d(self.inplanes),
							nn.ReLU(inplace=True),
							Encoding(D=512*self.expansion,K=16),
							nn.BatchNorm1d(16),
							nn.ReLU(inplace=True),
							nn2.View(-1, 512*self.expansion*16),
							nn.Linear(512*self.expansion*16, num_classes)]
		self.model = nn.Sequential(*model)
		print(model)

	def _residual_unit(self, block, planes, n_blocks, stride):
		strides = [stride] + [1]*(n_blocks-1)
		layers = []
		for i in range(n_blocks):
			layers += [block(self.inplanes, planes, strides[i])]
			self.inplanes = self.expansion*planes
		return nn.Sequential(*layers)

	def forward(self, input):
		return self.model(input)
