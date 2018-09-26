import os
import random
import scipy.io
import numpy as np
from PIL import Image, ImageOps, ImageFilter

from .base import BaseDataset

class VOCAugSegmentation(BaseDataset):
    voc = [
        'background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv'
    ]
    NUM_CLASS = 21
    TRAIN_BASE_DIR = 'VOCaug/dataset/'
    def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(VOCAugSegmentation, self).__init__(root, split, mode, transform,
                                                 target_transform, **kwargs)
        # train/val/test splits are pre-cut
        _voc_root = os.path.join(root, self.TRAIN_BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'cls')
        _image_dir = os.path.join(_voc_root, 'img')
        if self.mode == 'train':
            _split_f = os.path.join(_voc_root, 'trainval.txt')
        elif self.mode == 'val':
            _split_f = os.path.join(_voc_root, 'val.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n')+".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                if self.mode != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n')+".mat")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                _img = self.transform(_img)
            return _img, os.path.basename(self.images[index])
        _target = self._load_mat(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            _img, _target = self._sync_transform( _img, _target)
        elif self.mode == 'val':
            _img, _target = self._val_sync_transform( _img, _target)
        # general resize, normalize and toTensor
        if self.transform is not None:
            _img = self.transform(_img)
        if self.target_transform is not None:
            _target = self.target_transform(_target)
        return _img, _target
    
    def _load_mat(self, filename):
        mat = scipy.io.loadmat(filename, mat_dtype=True, squeeze_me=True, 
            struct_as_record=False)
        mask = mat['GTcls'].Segmentation
        return Image.fromarray(mask)

    def __len__(self):
        return len(self.images)
