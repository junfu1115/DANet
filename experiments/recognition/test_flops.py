##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse
import torch

from thop import profile, clever_format

import encoding

def get_args():
    # data settings
    parser = argparse.ArgumentParser(description='Deep Encoding')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='crop image size')
    # model params 
    parser.add_argument('--model', type=str, default='densenet',
                        help='network model type (default: densenet)')
    parser.add_argument('--rectify', action='store_true', 
                        default=False, help='rectify convolution')
    parser.add_argument('--rectify-avg', action='store_true', 
                        default=False, help='rectify convolution')
    # checking point
    parser = parser

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    model_kwargs = {}
    if args.rectify:
        model_kwargs['rectified_conv'] = True
        model_kwargs['rectify_avg'] = args.rectify_avg

    model = encoding.models.get_model(args.model, **model_kwargs)
    print(model)

    dummy_images = torch.rand(1, 3, args.crop_size, args.crop_size)

    #count_ops(model, dummy_images, verbose=False)
    macs, params = profile(model, inputs=(dummy_images, ))
    macs, params = clever_format([macs, params], "%.3f") 

    print(f"macs: {macs}, params: {params}")

if __name__ == '__main__':
    main()
