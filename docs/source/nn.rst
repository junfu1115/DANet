.. role:: hidden
    :class: hidden-section

Other NN Layers
===============

.. automodule:: encoding.nn

Customized Layers
-----------------

:hidden:`Normalize`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: Normalize
    :members:

:hidden:`View`
~~~~~~~~~~~~~~

.. autoclass:: View
    :members:

Standard Layers
---------------

Standard Layers as in PyTorch but in :class:`encoding.parallel.SelfDataParallel` mode. Use together with SyncBN.

:hidden:`Conv1d`
~~~~~~~~~~~~~~~~

.. autoclass:: Conv1d
    :members:

:hidden:`Conv2d`
~~~~~~~~~~~~~~~~

.. autoclass:: Conv2d
    :members:

:hidden:`ConvTranspose2d`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ConvTranspose2d
    :members:

:hidden:`ReLU`
~~~~~~~~~~~~~~

.. autoclass:: ReLU
    :members:

:hidden:`Sigmoid`
~~~~~~~~~~~~~~~~~

.. autoclass:: Sigmoid
    :members:

:hidden:`MaxPool2d`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: MaxPool2d
    :members:

:hidden:`AvgPool2d`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: AvgPool2d
    :members:

:hidden:`AdaptiveAvgPool2d`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AdaptiveAvgPool2d
    :members:

:hidden:`Dropout2d`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: Dropout2d
    :members:

:hidden:`Linear`
~~~~~~~~~~~~~~~~

.. autoclass:: Linear
    :members:


