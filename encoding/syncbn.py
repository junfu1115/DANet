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

