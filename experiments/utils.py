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
import shutil
import os
import sys
import time
import math

def adjust_learning_rate(optimizer, epoch, best_pred, args):
	lr = args.lr * ((0.1 ** int(epoch > 80)) * (0.1 ** int(epoch > 120)))
	print('=>Epoches %i, learning rate = %.4f, previous best = %.3f%%' % (
		epoch, lr, best_pred))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

def save_checkpoint(state, args, is_best, filename='checkpoint.pth.tar'):
	"""Saves checkpoint to disk"""
	directory = "runs/%s/%s/"%(args.dataset, args.checkname)
	if not os.path.exists(directory):
		os.makedirs(directory)
	filename = directory + filename
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, directory + 'model_best.pth.tar')

# taken from https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py
TOTAL_BAR_LENGTH = 86.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
	global last_time, begin_time
	if current == 0:
		begin_time = time.time()	# Reset for new bar.

	cur_len = int(TOTAL_BAR_LENGTH*current/total)
	rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

	sys.stdout.write(' [')
	for i in range(cur_len):
		sys.stdout.write('=')
	sys.stdout.write('>')
	for i in range(rest_len):
		sys.stdout.write('.')
	sys.stdout.write(']')

	cur_time = time.time()
	step_time = cur_time - last_time
	last_time = cur_time
	tot_time = cur_time - begin_time

	L = []
	L.append('	Step: %s' % format_time(step_time))
	L.append(' | Tot: %s' % format_time(tot_time))
	if msg:
		L.append(' | ' + msg)

	msg = ''.join(L)
	sys.stdout.write(msg)
	for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
		sys.stdout.write(' ')

	# Go back to the center of the bar.
	for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
		sys.stdout.write('\b')
	sys.stdout.write(' %d/%d ' % (current+1, total))

	if current < total-1:
		sys.stdout.write('\r')
	else:
		sys.stdout.write('\n')
	sys.stdout.flush()

def format_time(seconds):
	days = int(seconds / 3600/24)
	seconds = seconds - days*3600*24
	hours = int(seconds / 3600)
	seconds = seconds - hours*3600
	minutes = int(seconds / 60)
	seconds = seconds - minutes*60
	secondsf = int(seconds)
	seconds = seconds - secondsf
	millis = int(seconds*1000)

	f = ''
	i = 1
	if days > 0:
		f += str(days) + 'D'
		i += 1
	if hours > 0 and i <= 2:
		f += str(hours) + 'h'
		i += 1
	if minutes > 0 and i <= 2:
		f += str(minutes) + 'm'
		i += 1
	if secondsf > 0 and i <= 2:
		f += str(secondsf) + 's'
		i += 1
	if millis > 0 and i <= 2:
		f += str(millis) + 'ms'
		i += 1
	if f == '':
		f = '0ms'
	return f

