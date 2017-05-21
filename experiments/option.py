##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse
import os

class Options():
	def __init__(self):
		# Training settings
		parser = argparse.ArgumentParser(description='Deep Encoding')
		parser.add_argument('--dataset', type=str, default='cifar',
					help='training dataset (default: cifar)')
		parser.add_argument('--batch-size', type=int, default=128, metavar='N',
					help='input batch size for training (default: 64)')
		parser.add_argument('--test-batch-size', type=int, default=1000, 
				metavar='N', help='input batch size for testing (default: 1000)')
		parser.add_argument('--epochs', type=int, default=160, metavar='N',
					help='number of epochs to train (default: 10)')
		parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
					help='number of epochs to train (default: 10)')
		parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
					help='learning rate (default: 0.01)')
		parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
					help='SGD momentum (default: 0.5)')
		parser.add_argument('--no-cuda', action='store_true', default=False,
					help='disables CUDA training')
		parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
		parser.add_argument('--log-interval', type=int, default=10, metavar=
				'N',help='how many batches to wait before logging status')	
		parser.add_argument('--resume', type=str, default=None,
					help='put the path to resuming file if needed')
		parser.add_argument('--checkname', type=str, default='default',
					help='set the checkpoint name')
		self.parser = parser
	def parse(self):
		return self.parser.parse_args()
