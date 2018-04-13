.. role:: hidden
    :class: hidden-section

Data Parallel
=============

- Current PyTorch DataParallel Table is not supporting mutl-gpu loss calculation, which makes the gpu memory usage very in-balance. We address this issue here by doing Model & CriterionDataParallel. 

.. note::
    This code is provided together with the paper

    * Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, Amit Agrawal. "Context Encoding for Semantic Segmentation"  *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*


.. automodule:: encoding.parallel
.. currentmodule:: encoding.parallel

:hidden:`ModelDataParallel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ModelDataParallel
    :members:

:hidden:`CriterionDataParallel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CriterionDataParallel
    :members:


:hidden:`allreduce`
~~~~~~~~~~~~~~~~~~~

.. autofunction:: allreduce
