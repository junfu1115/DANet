##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Customized functions"""

import torch
from torch.autograd import Variable, Function
from .. import lib

__all__ = ['NonMaxSuppression']

def NonMaxSuppression(boxes, scores, threshold):
    r"""Non-Maximum Suppression
    The algorithm begins by storing the highest-scoring bounding
    box, and eliminating any box whose intersection-over-union (IoU)
    with it is too great. The procedure repeats on the surviving
    boxes, and so on until there are no boxes left.
    The stored boxes are returned.

    NB: The function returns a tuple (mask, indices), where
    indices index into the input boxes and are sorted
    according to score, from higest to lowest.
    indices[i][mask[i]] gives the indices of the surviving
    boxes from the ith batch, sorted by score.

    Args:
      - boxes :math:`(N, n_boxes, 4)`
      - scroes :math:`(N, n_boxes)`
      - threshold (float): IoU above which to eliminate boxes

    Outputs:
      - mask: :math:`(N, n_boxes)`
      - indicies: :math:`(N, n_boxes)`

    Examples::

    >>> boxes = torch.Tensor([[[10., 20., 20., 15.],
    >>>                       [24., 22., 50., 54.],
    >>>                       [10., 21., 20. 14.5]]])
    >>> scores = torch.abs(torch.randn([1, 3]))
    >>> mask, indices = NonMaxSuppression(boxes, scores, 0.7)
    >>> #indices are SORTED according to score.
    >>> surviving_box_indices = indices[mask]
    """
    if boxes.is_cuda:
        return lib.gpu.non_max_suppression(boxes, scores, threshold)
    else:
        return lib.cpu.non_max_suppression(boxes, scores, threshold)
