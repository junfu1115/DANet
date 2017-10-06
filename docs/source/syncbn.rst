.. role:: hidden
    :class: hidden-section

Synchronized BatchNorm
======================

The current BN is implementated insynchronized accross the gpus, which is a big problem for memory consuming tasks such as Semantic Segmenation, since the mini-batch is very small. 
To synchronize the batchnorm accross multiple gpus is not easy to implment within the current Dataparallel framework. We address this difficulty by making each layer 'self-parallel', that is accepting the inputs from multi-gpus. Therefore, we can handle different layers seperately for synchronizing it across gpus.

.. currentmodule:: encoding

Functions
---------

:hidden:`batchnormtrain`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: batchnormtrain
    :members:

:hidden:`batchnormeval`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: batchnormeval
    :members:

:hidden:`sum_square`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sum_square
    :members:

