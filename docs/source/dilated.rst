.. role:: hidden
    :class: hidden-section

Dilated Networks
================

We provide correct dilated pre-trained ResNet and DenseNet for semantic segmentation. 
For dilation of ResNet, we replace the stride of 2 Conv3x3 at begining of certain stage and update the dilation of the conv layers afterwards. 
For dilation of DenseNet, we provide DilatedAvgPool2d that handles the dilation of the transition layers, then update the dilation of the conv layers afterwards. 
All provided models have been verified. 


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


DenseNet
--------

:hidden:`DenseNet`
~~~~~~~~~~~~~~~~~~

.. autoclass:: DenseNet
    :members:


:hidden:`densenet161`
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: densenet161


:hidden:`densenet121`
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: densenet121


:hidden:`densenet169`
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: densenet169


:hidden:`densenet201`
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: densenet201


