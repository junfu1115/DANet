.. role:: hidden
    :class: hidden-section

Synchronized BatchNorm
======================

The current BN is implementated insynchronized accross the gpus, which is a big problem for memory consuming tasks such as Semantic Segmenation, since the mini-batch is very small. 
To synchronize the batchnorm accross multiple gpus is not easy to implment within the current Dataparallel framework. We address this difficulty by making each layer 'self-parallel' :class:`encoding.parallel.SelfDataParallel`, that is accepting the inputs from multi-gpus. Therefore, we can handle the synchronizing across gpus.


.. currentmodule:: encoding.nn

Modules
-------

:hidden:`BatchNorm1d`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BatchNorm1d
    :members:

:hidden:`BatchNorm2d`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BatchNorm2d
    :members:


.. currentmodule:: encoding


Functions
---------

.. currentmodule:: encoding.functions


:hidden:`batchnormtrain`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: batchnormtrain

:hidden:`batchnormeval`
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: batchnormeval

:hidden:`sum_square`
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sum_square

