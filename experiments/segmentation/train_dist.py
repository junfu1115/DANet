##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
from torch.nn.parallel import DistributedDataParallel

import encoding.utils as utils
from encoding.nn import SegmentationLosses, DistSyncBatchNorm

from encoding.datasets import get_dataset
from encoding.models import get_segmentation_model


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch \
            Segmentation')
        # model and dataset 
        parser.add_argument('--model', type=str, default='encnet',
                            help='model name (default: encnet)')
        parser.add_argument('--backbone', type=str, default='resnet50',
                            help='backbone name (default: resnet50)')
        parser.add_argument('--rectify', action='store_true', 
                            default=False, help='rectify convolution')
        parser.add_argument('--rectify-avg', action='store_true', 
                            default=False, help='rectify convolution')
        parser.add_argument('--dataset', type=str, default='ade20k',
                            help='dataset name (default: pascal12)')
        parser.add_argument('--workers', type=int, default=8,
                            metavar='N', help='dataloader threads')
        parser.add_argument('--base-size', type=int, default=520,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=480,
                            help='crop image size')
        parser.add_argument('--train-split', type=str, default='train',
                            help='dataset train split (default: train)')
        # training hyper params
        parser.add_argument('--aux', action='store_true', default= False,
                            help='Auxilary Loss')
        parser.add_argument('--aux-weight', type=float, default=0.2,
                            help='Auxilary loss weight (default: 0.2)')
        parser.add_argument('--se-loss', action='store_true', default= False,
                            help='Semantic Encoding Loss SE-loss')
        parser.add_argument('--se-weight', type=float, default=0.2,
                            help='SE-loss weight (default: 0.2)')
        parser.add_argument('--epochs', type=int, default=None, metavar='N',
                            help='number of epochs to train (default: auto)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='start epochs (default:0)')
        parser.add_argument('--batch-size', type=int, default=2,
                            metavar='N', help='input batch size for \
                            training (default: auto)')
        parser.add_argument('--test-batch-size', type=int, default=16,
                            metavar='N', help='input batch size for \
                            testing (default: same as batch size)')
        # optimizer params
        parser.add_argument('--lr', type=float, default=None, metavar='LR',
                            help='learning rate (default: auto)')
        parser.add_argument('--lr-scheduler', type=str, default='poly',
                            help='learning rate scheduler (default: poly)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=1e-4,
                            metavar='M', help='w-decay (default: 1e-4)')
        # cuda, seed and logging
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default='default',
                            help='set the checkpoint name')
        parser.add_argument('--model-zoo', type=str, default=None,
                            help='evaluating on model zoo model')
        # finetuning pre-trained models
        parser.add_argument('--ft', action='store_true', default= False,
                            help='finetuning on a different dataset')
        # evaluation option
        parser.add_argument('--eval', action='store_true', default= False,
                            help='evaluating mIoU')
        parser.add_argument('--export', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--test-val', action='store_true', default= False,
                            help='generate masks on val set')
        # test option
        parser.add_argument('--test-folder', type=str, default=None,
                            help='path to test image folder')
        # distributed
        parser.add_argument('--world-size', default=1, type=int,
                            help='number of nodes for distributed training')
        parser.add_argument('--rank', default=0, type=int,
                            help='node rank for distributed training')
        parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        # default settings for epochs, batch_size and lr
        if args.epochs is None:
            epoches = {
                'coco': 30,
                'pascal_aug': 80,
                'pascal_voc': 50,
                'pcontext': 80,
                'ade20k': 120,
                'citys': 240,
            }
            args.epochs = epoches[args.dataset.lower()]
        if args.lr is None:
            lrs = {
                'coco': 0.004,
                'pascal_aug': 0.001,
                'pascal_voc': 0.0001,
                'pcontext': 0.001,
                'ade20k': 0.01,
                'citys': 0.01,
            }
            args.lr = lrs[args.dataset.lower()] / 16 * args.batch_size
        print(args)
        return args

def main():
    args = Options().parse()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    args.lr = args.lr * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

best_pred = 0.0

def main_worker(gpu, ngpus_per_node, args):
    global best_pred
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    print('rank: {} / {}'.format(args.rank, args.world_size))
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    # dataset
    data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                   'crop_size': args.crop_size}
    trainset = get_dataset(args.dataset, split=args.train_split, mode='train', **data_kwargs)
    valset = get_dataset(args.dataset, split='val', mode ='val', **data_kwargs)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(valset, shuffle=False)
    # dataloader
    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': args.workers, 'pin_memory': True}
    trainloader = data.DataLoader(trainset, sampler=train_sampler, drop_last=True, **loader_kwargs)
    valloader = data.DataLoader(valset, sampler=val_sampler, **loader_kwargs)
    nclass = trainset.num_class
    # model
    model_kwargs = {}
    if args.rectify:
        model_kwargs['rectified_conv'] = True
        model_kwargs['rectify_avg'] = args.rectify_avg
    model = get_segmentation_model(args.model, dataset=args.dataset,
                                   backbone=args.backbone, aux=args.aux,
                                   se_loss=args.se_loss, norm_layer=DistSyncBatchNorm,
                                   base_size=args.base_size, crop_size=args.crop_size,
                                   **model_kwargs)
    if args.gpu == 0:
        print(model)
    # optimizer using different LR
    params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},]
    if hasattr(model, 'head'):
        params_list.append({'params': model.head.parameters(), 'lr': args.lr*10})
    if hasattr(model, 'auxlayer'):
        params_list.append({'params': model.auxlayer.parameters(), 'lr': args.lr*10})
    optimizer = torch.optim.SGD(params_list,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # criterions
    criterion = SegmentationLosses(se_loss=args.se_loss,
                                   aux=args.aux,
                                   nclass=nclass, 
                                   se_weight=args.se_weight,
                                   aux_weight=args.aux_weight)
    # distributed data parallel
    model.cuda(args.gpu)
    criterion.cuda(args.gpu)
    model = DistributedDataParallel(model, device_ids=[args.gpu])
    metric = utils.SegmentationMetric(nclass=nclass)

    # resuming checkpoint
    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        if not args.ft:
            optimizer.load_state_dict(checkpoint['optimizer'])
        best_pred = checkpoint['best_pred']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    # clear start epoch if fine-tuning
    if args.ft:
        args.start_epoch = 0

    # lr scheduler
    scheduler = utils.LR_Scheduler_Head(args.lr_scheduler, args.lr,
                                        args.epochs, len(trainloader))

    def training(epoch):
        train_sampler.set_epoch(epoch)
        global best_pred
        train_loss = 0.0
        model.train()
        tic = time.time()
        for i, (image, target) in enumerate(trainloader):
            scheduler(optimizer, i, epoch, best_pred)
            optimizer.zero_grad()
            outputs = model(image)
            target = target.cuda(args.gpu)
            loss = criterion(*outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if i % 100 == 0 and args.gpu == 0:
                iter_per_sec = 100.0 / (time.time() - tic) if i != 0 else 1.0/ (time.time() - tic)
                tic = time.time()
                print('Epoch: {}, Iter: {}, Speed: {:.3f} iter/sec, Train loss: {:.3f}'. \
                      format(epoch, i, iter_per_sec, train_loss / (i + 1)))

    def validation(epoch):
        # Fast test during the training using single-crop only
        global best_pred
        is_best = False
        model.eval()
        metric.reset()

        for i, (image, target) in enumerate(valloader):
            with torch.no_grad():
                pred = model(image)[0]
                target = target.cuda(args.gpu)
                metric.update(target, pred)

            if i % 100 == 0:
                all_metircs = metric.get_all()
                all_metircs = utils.torch_dist_sum(args.gpu, *all_metircs)
                pixAcc, mIoU = utils.get_pixacc_miou(*all_metircs)
                if args.gpu == 0:
                    print('pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))

        all_metircs = metric.get_all()
        all_metircs = utils.torch_dist_sum(args.gpu, *all_metircs)
        pixAcc, mIoU = utils.get_pixacc_miou(*all_metircs)
        if args.gpu == 0:
            print('pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))
            if args.eval: return
            new_pred = (pixAcc + mIoU)/2
            if new_pred > best_pred:
                is_best = True
                best_pred = new_pred
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_pred': best_pred,
            }, args, is_best)

    if args.export:
        if args.gpu == 0:
            torch.save(model.module.state_dict(), args.export + '.pth')
        return

    if args.eval:
        validation(args.start_epoch)
        return

    if args.gpu == 0:
        print('Starting Epoch:', args.start_epoch)
        print('Total Epoches:', args.epochs)

    for epoch in range(args.start_epoch, args.epochs):
        tic = time.time()
        training(epoch)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            validation(epoch)
        elapsed = time.time() - tic
        if args.gpu == 0:
            print(f'Epoch: {epoch}, Time cost: {elapsed}')

    #validation(epoch)


if __name__ == "__main__":
    main()
