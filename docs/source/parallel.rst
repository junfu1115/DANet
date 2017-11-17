.. role:: hidden
    :class: hidden-section

Data Parallel
=============

- Current PyTorch DataParallel Table is not supporting mutl-gpu loss calculation, which makes the gpu memory usage very in-efficient. We address this issue here by doing CriterionDataParallel. 
- :class:`encoding.parallel.SelfDataParallel` is compatible with Synchronized Batch Normalization :class:`encoding.nn.BatchNorm2d`.

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

:hidden:`SelfDataParallel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SelfDataParallel
    :members:

:hidden:`AllReduce`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AllReduce
    :members:

:hidden:`Broadcast`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Broadcast
    :members:

