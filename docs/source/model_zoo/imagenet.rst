Image Classification
====================

Install Package
---------------

- Clone the GitHub repo::
    
    git clone https://github.com/zhanghang1989/PyTorch-Encoding

- Install PyTorch Encoding (if not yet). Please follow the installation guide `Installing PyTorch Encoding <../notes/compile.html>`_.

Get Pre-trained Model
---------------------

.. hint::
    How to get pretrained model, for example ``ResNeSt50``::

        model = encoding.models.get_model('ResNeSt50', pretrained=True)

    After clicking ``cmd`` in the table, the command for training the model can be found below the table.

.. role:: raw-html(raw)
   :format: html


ResNeSt
~~~~~~~

.. note::
    The provided models were trained using MXNet Gluon, this PyTorch implementation is slightly worse than the original implementation.

===============================  ==============    ==============    =========================================================================================================
Model                            crop-size         Acc               Command                                                                                      
===============================  ==============    ==============    =========================================================================================================
ResNeSt-50                       224               81.03             :raw-html:`<a href="javascript:toggleblock('cmd_resnest50')" class="toggleblock">cmd</a>`
ResNeSt-101                      256               82.83             :raw-html:`<a href="javascript:toggleblock('cmd_resnest101')" class="toggleblock">cmd</a>`
ResNeSt-200                      320               83.84             :raw-html:`<a href="javascript:toggleblock('cmd_resnest200')" class="toggleblock">cmd</a>`
ResNeSt-269                      416               84.54             :raw-html:`<a href="javascript:toggleblock('cmd_resnest269')" class="toggleblock">cmd</a>`
===============================  ==============    ==============    =========================================================================================================

.. raw:: html

    <code xml:space="preserve" id="cmd_resnest50" style="display: none; text-align: left; white-space: pre-wrap">
    # change the rank for worker node
    python train_dist.py --dataset imagenet --model resnest50 --lr-scheduler cos --epochs 270 --checkname resnest50 --lr 0.025 --batch-size 64 --dist-url tcp://MASTER:NODE:IP:ADDRESS:23456 --world-size 4 --label-smoothing 0.1 --mixup 0.2 --no-bn-wd --last-gamma --warmup-epochs 5 --rand-aug --rank 0
    </code>

    <code xml:space="preserve" id="cmd_resnest101" style="display: none; text-align: left; white-space: pre-wrap">
    # change the rank for worker node
    python train_dist.py --dataset imagenet --model resnest101 --lr-scheduler cos --epochs 270 --checkname resnest101 --lr 0.025 --batch-size 64 --dist-url tcp://MASTER:NODE:IP:ADDRESS:23456 --world-size 4 --label-smoothing 0.1 --mixup 0.2 --no-bn-wd --last-gamma --warmup-epochs 5 --rand-aug --rank 0
    </code>

    <code xml:space="preserve" id="cmd_resnest200" style="display: none; text-align: left; white-space: pre-wrap">
    # change the rank for worker node
    python train_dist.py --dataset imagenet --model resnest200 --lr-scheduler cos --epochs 270 --checkname resnest200 --lr 0.0125 --batch-size 32 --dist-url tcp://MASTER:NODE:IP:ADDRESS:23456 --world-size 8 --label-smoothing 0.1 --mixup 0.2 --no-bn-wd --last-gamma --warmup-epochs 5 --rand-aug --crop-size 256 --rank 0
    </code>

    <code xml:space="preserve" id="cmd_resnest269" style="display: none; text-align: left; white-space: pre-wrap">
    # change the rank for worker node
    python train_dist.py --dataset imagenet --model resnest269 --lr-scheduler cos --epochs 270 --checkname resnest269 --lr 0.0125 --batch-size 32 --dist-url tcp://MASTER:NODE:IP:ADDRESS:23456 --world-size 8 --label-smoothing 0.1 --mixup 0.2 --no-bn-wd --last-gamma --warmup-epochs 5 --rand-aug --crop-size 320 --rank 0
    </code>

Test Pretrained
~~~~~~~~~~~~~~~

- Prepare the datasets by downloading the data into current folder and then runing the scripts in the ``scripts/`` folder::

      python scripts/prepare_imagenet.py --data-dir ./
  
- The test script is in the ``experiments/recognition/`` folder. For evaluating the model (using MS),
  for example ``ResNeSt50``::

      python verify.py --dataset imagenet --model ResNeSt50 --crop-size 224

Train Your Own Model
--------------------

- Prepare the datasets by downloading the data into current folder and then runing the scripts in the ``scripts/`` folder::

    python scripts/prepare_imagenet.py --data-dir ./

- The training script is in the ``experiments/recognition/`` folder. Commands for reproducing pre-trained models can be found in the table.


