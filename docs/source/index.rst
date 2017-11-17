.. Encoding documentation master file

:github_url: https://github.com/zhanghang1989/PyTorch-Encoding

Encoding Documentation
======================

Created by `Hang Zhang <http://hangzh.com/>`_

- An optimized PyTorch package with CUDA backend, including Encoding Layer :class:`encoding.nn.Encoding`, Multi-GPU Synchronized Batch Normalization :class:`encoding.nn.BatchNorm2d` and other customized modules and functions. 

- **Example Systems** for Semantic Segmentation (coming), CIFAR-10 Classification, `Texture Recognition <experiments/texture.html>`_ and `Style Transfer <experiments/style.html>`_ are provided in experiments section. 


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Notes

   notes/*

.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   encoding
   syncbn
   parallel
   dilated
   nn
   functions
   utils

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Experiment Systems

   experiments/*

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
