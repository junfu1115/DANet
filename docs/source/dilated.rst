.. role:: hidden
    :class: hidden-section

encoding.dilated
================

We provide correct dilated pre-trained ResNet and DenseNet (stride of 8) for semantic segmentation. 
For dilation of DenseNet, we provide :class:`encoding.nn.DilatedAvgPool2d`. 
All provided models have been verified. 

.. note::
    This code is provided together with the paper

    * Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, Amit Agrawal. "Context Encoding for Semantic Segmentation"  *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*


.. automodule:: encoding.dilated
.. currentmodule:: encoding.dilated

ResNet
------

:hidden:`ResNet`
~~~~~~~~~~~~~~~~

.. autoclass:: ResNet
    :members:

:hidden:`resnet18`
~~~~~~~~~~~~~~~~~~

.. autofunction:: resnet18

:hidden:`resnet34`
~~~~~~~~~~~~~~~~~~

.. autofunction:: resnet34

:hidden:`resnet50`
~~~~~~~~~~~~~~~~~~

.. autofunction:: resnet50

:hidden:`resnet101`
~~~~~~~~~~~~~~~~~~~

.. autofunction:: resnet101

:hidden:`resnet152`
~~~~~~~~~~~~~~~~~~~

.. autofunction:: resnet152
