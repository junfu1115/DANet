###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn 
# Copyright (c) 2018
###########################################################################

import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses, BatchNorm2d
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_segmentation_dataset, test_batchify_fn
from encoding.models import get_model, get_segmentation_model, MultiEvalModule

from option import Options
torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

def test(args):
    # output folder
    outdir = '%s/danet_vis'%(args.dataset)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    # dataset
    if args.eval:
        testset = get_segmentation_dataset(args.dataset, split='val', mode='testval',
                                           transform=input_transform)
    else:#set split='test' for test set
        testset = get_segmentation_dataset(args.dataset, split='val', mode='vis',
                                           transform=input_transform)
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False,
                                collate_fn=test_batchify_fn, **loader_kwargs)
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone=args.backbone, aux=args.aux,
                                       se_loss=args.se_loss, norm_layer=BatchNorm2d,
                                       base_size=args.base_size, crop_size=args.crop_size,
                                       multi_grid=args.multi_grid, multi_dilation=args.multi_dilation)
        # resuming checkpoint
        if args.resume is None or not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume)
        # strict=False, so that it is compatible with old pytorch saved models
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    print(model)
    num_class = testset.num_class
    evaluator = MultiEvalModule(model, testset.num_class, multi_scales=args.multi_scales).cuda()
    evaluator.eval()

    tbar = tqdm(test_data)
    def eval_batch(image, dst, evaluator, eval_mode):
        if eval_mode:
            # evaluation mode on validation set
            targets = dst
            outputs = evaluator.parallel_forward(image)
            
            batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
            for output, target in zip(outputs, targets):
                correct, labeled = utils.batch_pix_accuracy(output.data.cpu(), target)
                inter, union = utils.batch_intersection_union(
                    output.data.cpu(), target, testset.num_class)
                batch_correct += correct
                batch_label += labeled
                batch_inter += inter
                batch_union += union
            return batch_correct, batch_label, batch_inter, batch_union
        else:
            # Visualize and dump the results
            im_paths = dst
            outputs = evaluator.parallel_forward(image)
            predicts = [torch.max(output, 1)[1].cpu().numpy() + testset.pred_offset
                        for output in outputs]
            for predict, impath in zip(predicts, im_paths):
                mask = utils.get_mask_pallete(predict, args.dataset)
                outname = os.path.splitext(impath)[0] + '.png'
                mask.save(os.path.join(outdir, outname))
            # dummy outputs for compatible with eval mode
            return 0, 0, 0, 0

    total_inter, total_union, total_correct, total_label = \
        np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    for i, (image, dst) in enumerate(tbar):
        if torch_ver == "0.3":
            image = Variable(image, volatile=True)
            correct, labeled, inter, union = eval_batch(image, dst, evaluator, args.eval)
        else:
            with torch.no_grad():
                correct, labeled, inter, union = eval_batch(image, dst, evaluator, args.eval)
        pixAcc, mIoU, IoU = 0, 0, 0
        if args.eval:
            total_correct += correct.astype('int64')
            total_label += labeled.astype('int64')
            total_inter += inter.astype('int64')
            total_union += union.astype('int64')
            pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
            IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
            mIoU = IoU.mean()
            tbar.set_description(
                'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
    return pixAcc, mIoU, IoU, num_class

def eval_multi_models(args):
    if args.resume_dir is None or not os.path.isdir(args.resume_dir):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume_dir))
    for resume_file in os.listdir(args.resume_dir):
        if os.path.splitext(resume_file)[1] == '.tar':
            args.resume = os.path.join(args.resume_dir, resume_file)
            assert os.path.exists(args.resume)
            if not args.eval:
                test(args)
                continue
            pixAcc, mIoU, IoU, num_class = test(args)
        
            txtfile = args.resume
            txtfile = txtfile.replace('pth.tar', 'txt')
            if not args.multi_scales:
                txtfile = txtfile.replace('.txt', 'result_mIoU_%.4f.txt'%mIoU)
            else:
                txtfile = txtfile.replace('.txt', 'multi_scale_result_mIoU_%.4f.txt'%mIoU)
            fh = open(txtfile, 'w')
            print("================ Summary IOU ================\n")
            for i in range(0,num_class):
                print("%3d: %.4f\n" %(i,IoU[i]))
                fh.write("%3d: %.4f\n" %(i,IoU[i]))
            print("Mean IoU over %d classes: %.4f\n" % (num_class, mIoU))
            print("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
            fh.write("Mean IoU over %d classes: %.4f\n" % (num_class, mIoU))
            fh.write("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
            fh.close()
    print('Evaluation is finished!!!')

if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    eval_multi_models(args)
