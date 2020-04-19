import torch
import numpy as np
from encoding.utils.metrics import *

def test_segmentation_metrics():
    # check torch evaluation metrics
    rows, cols = 640, 480
    nclass = 30
    # numpy data
    im_lab = np.matrix(np.random.randint(0, nclass, size=(rows, cols)))
    mask = np.random.random((nclass, rows, cols))
    im_pred = mask.argmax(axis=0)
    # torch data
    tim_lab = torch.from_numpy(im_lab).unsqueeze(0).long()
    tim_pred = torch.from_numpy(mask).unsqueeze(0)
    # numpy prediction
    pixel_correct, pixel_labeled = pixel_accuracy(im_pred, im_lab)
    area_inter, area_union = intersection_and_union(im_pred, im_lab, nclass)
    pixAcc = 1.0 * pixel_correct / (np.spacing(1) + pixel_labeled)
    IoU = 1.0 * area_inter / (np.spacing(1) + area_union)
    mIoU = IoU.mean()
    print('numpy predictionis :', pixAcc, mIoU)
    # torch metric prediction
    pixel_correct, pixel_labeled = batch_pix_accuracy(tim_pred, tim_lab)
    area_inter, area_union = batch_intersection_union(tim_pred, tim_lab, nclass)
    batch_pixAcc = 1.0 * pixel_correct / (np.spacing(1) + pixel_labeled)
    IoU = 1.0 * area_inter / (np.spacing(1) + area_union)
    batch_mIoU = IoU.mean()
    print('torch predictionis :', batch_pixAcc, batch_mIoU)
    assert (batch_pixAcc - pixAcc) < 1e-3
    assert (batch_mIoU - mIoU) < 1e-3
