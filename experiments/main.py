##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from option import Options
from model.encodenet import Net
from utils import *

# global variable
best_pred = 0.0
acclist = []

def main():
	# init the args
	args = Options().parse()
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)
	# init dataloader
	if args.dataset == 'cifar':
		from dataset.cifar import Dataloder
		train_loader, test_loader = Dataloder(args).getloader()
	else:
		raise ValueError('Unknow dataset!')

	model = Net()

	if args.cuda:
		model.cuda()

	if args.resume is not None:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_pred = checkpoint['best_pred']
			acclist = checkpoint['acclist']
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})"
				.format(args.resume, checkpoint['epoch']))
		else:
			print("=> no resume checkpoint found at '{}'".format(args.resume))

	criterion = nn.CrossEntropyLoss()
	# TODO make weight_decay oen of args
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=
			args.momentum, weight_decay=1e-4)

	def train(epoch):
		model.train()
		global best_pred
		train_loss, correct, total = 0,0,0
		adjust_learning_rate(optimizer, epoch, best_pred, args)
		for batch_idx, (data, target) in enumerate(train_loader):
			if args.cuda:
				data, target = data.cuda(), target.cuda()
			data, target = Variable(data), Variable(target)
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()

			train_loss += loss.data[0]
			pred = output.data.max(1)[1] 
			correct += pred.eq(target.data).cpu().sum()
			total += target.size(0)
			progress_bar(batch_idx, len(train_loader), 
				'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 
				100.*correct/total, correct, total))

	def test(epoch):
		model.eval()
		global best_pred
		global acclist
		test_loss, correct, total = 0,0,0
		acc = 0.0
		is_best = False
		# for data, target in test_loader:
		for batch_idx, (data, target) in enumerate(test_loader):
			if args.cuda:
				data, target = data.cuda(), target.cuda()
			data, target = Variable(data, volatile=True), Variable(target)
			output = model(data)
			test_loss += criterion(output, target).data[0]
			# get the index of the max log-probability
			pred = output.data.max(1)[1] 
			correct += pred.eq(target.data).cpu().sum()
			total += target.size(0)

			acc = 100.*correct/total
			progress_bar(batch_idx, len(test_loader), 
				'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 
				acc, correct, total))
		# save checkpoint
		acclist += [acc]
		if acc > best_pred:
			best_pred = acc
			is_best = True
		save_checkpoint({
			'epoch': epoch,
			'state_dict': model.state_dict(),
			'best_pred': best_pred,
			'acclist':acclist,
			}, args=args, is_best=is_best)

	# TODO add plot curve

	for epoch in range(args.start_epoch, args.epochs + 1):
		train(epoch)
		# FIXME this is a bug somewhere not in the code
		test(epoch)


if __name__ == "__main__":
	main()
