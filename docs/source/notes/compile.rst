Install PyTorch-Encoding
========================

- Install PyTorch from Source to the ``$HOME`` directory:
    * Please follow the `PyTorch instructions <https://github.com/pytorch/pytorch#from-source>`_ (recommended).
    * Or you can simply clone a copy to ``$HOME`` directory::

        git clone https://github.com/pytorch/pytorch $HOME/pytorch

- Install this package:
    * Clone the repo::

        git clone https://github.com/zhanghang1989/PyTorch-Encoding && cd PyTorch-Encoding-Layer

    * On Linux::

        python setup.py install

    * On MAC OSX::

         MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install

