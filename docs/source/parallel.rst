.. role:: hidden
    :class: hidden-section

encoding.parallel
=================

- Current PyTorch DataParallel Table is not supporting mutl-gpu loss calculation, which makes the gpu memory usage very in-balance. We address this issue here by doing DataParallel for Model & Criterion. 

.. note::
    Deprecated, please use torch.nn.parallel.DistributedDataParallel with :class:`encoding.nn.DistSyncBatchNorm` for the best performance.

.. automodule:: encoding.parallel
.. currentmodule:: encoding.parallel

:hidden:`DataParallelModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DataParallelModel
    :members:

:hidden:`DataParallelCriterion`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DataParallelCriterion
    :members:


:hidden:`allreduce`
~~~~~~~~~~~~~~~~~~~

.. autofunction:: allreduce
