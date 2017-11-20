Installing PyTorch-Encoding
===========================


Install from Source
-------------------

    * Install PyTorch from Source (recommended). Please follow the `PyTorch instructions <https://github.com/pytorch/pytorch#from-source>`_.

    * Install this package

        - Clone the repo::

            git clone https://github.com/zhanghang1989/PyTorch-Encoding && cd PyTorch-Encoding

        - On Linux::

            python setup.py install

        - On Mac OSX::

             MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install

Reference
---------

    .. note::
        If using the code in your research, please cite our paper.

        * Hang Zhang, Jia Xue, and Kristin Dana. "Deep TEN: Texture Encoding Network." *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2017*::

            @InProceedings{Zhang_2017_CVPR,
            author = {Zhang, Hang and Xue, Jia and Dana, Kristin},
            title = {Deep TEN: Texture Encoding Network},
            booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
            month = {July},
            year = {2017}
            }
