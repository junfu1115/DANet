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
from torch.autograd import gradcheck

# declare dims and variables 
B, N, K, D = 1, 2, 3, 4
A = Variable(torch.randn(B,N,K).cuda(), requires_grad=True)
R = Variable(torch.randn(B,N,K,D).cuda(), requires_grad=True)

# check Aggregate operation
test = gradcheck(Aggregate(),(A, R), eps=1e-4, atol=1e-3)
print('Gradcheck of Aggreate() returns ', test)


