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
from encoding import Aggregate

model = Aggregate()

B, N, K, D = 1, 2, 3, 4
# TODO cpu test
A = Variable(torch.ones(B,N,K).cuda())
R = Variable(torch.ones(B,N,K,D).cuda())

E = model(A, R)
print(E)
