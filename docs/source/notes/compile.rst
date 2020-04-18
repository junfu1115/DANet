Install and Citations
=====================


Installation
------------

    * Install PyTorch 1.4.0 by following the `PyTorch instructions <http://pytorch.org/>`_.
 
    * PIP Install::

        pip install torch-encoding --pre

    * Install from source:: 

        git clone https://github.com/zhanghang1989/PyTorch-Encoding && cd PyTorch-Encoding
        python setup.py install


Detailed Steps
--------------

This tutorial is a sucessful setup example for AWS EC2 p3 instance with ubuntu 16.04, CUDA 10.
We cannot guarantee it to work for all the machines, but the steps should be similar.
Assuming CUDA and cudnn are already sucessfully installed, otherwise please refer to other tutorials.

      * Install Anaconda from the `link <https://www.anaconda.com/distribution/>`_ .

      * Install ninja::
 
         wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
         sudo unzip ninja-linux.zip -d /usr/local/bin/
         sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force

      * Install PyTorch::

         conda install pytorch torchvision cudatoolkit=100 -c pytorch

      * Install this package::

         pip install torch-encoding --pre

Citations
---------

    .. note::
        If using the code in your research, please cite our papers.

        * Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, Amit Agrawal. "Context Encoding for Semantic Segmentation"  *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*::

            @InProceedings{Zhang_2018_CVPR,
            author = {Zhang, Hang and Dana, Kristin and Shi, Jianping and Zhang, Zhongyue and Wang, Xiaogang and Tyagi, Ambrish and Agrawal, Amit},
            title = {Context Encoding for Semantic Segmentation},
            booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
            month = {June},
            year = {2018}
            }


        * Hang Zhang, Jia Xue, and Kristin Dana. "Deep TEN: Texture Encoding Network." *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2017*::

            @InProceedings{Zhang_2017_CVPR,
            author = {Zhang, Hang and Xue, Jia and Dana, Kristin},
            title = {Deep TEN: Texture Encoding Network},
            booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
            month = {July},
            year = {2017}
            }
