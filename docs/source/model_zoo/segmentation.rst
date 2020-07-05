Semantic Segmentation
=====================

Install Package
---------------

- Clone the GitHub repo::
    
    git clone https://github.com/zhanghang1989/PyTorch-Encoding

- Install PyTorch Encoding (if not yet). Please follow the installation guide `Installing PyTorch Encoding <../notes/compile.html>`_.

Get Pre-trained Model
---------------------

.. hint::
    The model names contain the training information. For instance ``EncNet_ResNet50s_ADE``:
      - ``EncNet`` indicate the algorithm is “Context Encoding for Semantic Segmentation”
      - ``ResNet50`` is the name of backbone network.
      - ``ADE`` means the ADE20K dataset.

    How to get pretrained model, for example ``EncNet_ResNet50s_ADE``::

        model = encoding.models.get_model('EncNet_ResNet50s_ADE', pretrained=True)

    After clicking ``cmd`` in the table, the command for training the model can be found below the table.

.. role:: raw-html(raw)
   :format: html


ResNeSt Backbone Models
-----------------------

ADE20K Dataset
~~~~~~~~~~~~~~

==============================================================================  ====================    ===================    =========================================================================================================
Model                                                                           pixAcc                  mIoU                   Command                                                                                      
==============================================================================  ====================    ===================    =========================================================================================================
FCN_ResNeSt50_ADE                                                               80.18%                  42.94%                 :raw-html:`<a href="javascript:toggleblock('cmd_fcn_nest50_ade')" class="toggleblock">cmd</a>`
DeepLab_ResNeSt50_ADE                                                           81.17%                  45.12%                 :raw-html:`<a href="javascript:toggleblock('cmd_deeplab_resnest50_ade')" class="toggleblock">cmd</a>`
DeepLab_ResNeSt101_ADE                                                          82.07%                  46.91%                 :raw-html:`<a href="javascript:toggleblock('cmd_deeplab_resnest101_ade')" class="toggleblock">cmd</a>`
DeepLab_ResNeSt200_ADE                                                          82.45%                  48.36%                 :raw-html:`<a href="javascript:toggleblock('cmd_deeplab_resnest200_ade')" class="toggleblock">cmd</a>`
DeepLab_ResNeSt269_ADE                                                          82.62%                  47.60%                 :raw-html:`<a href="javascript:toggleblock('cmd_deeplab_resnest269_ade')" class="toggleblock">cmd</a>`
==============================================================================  ====================    ===================    =========================================================================================================

.. raw:: html

    <code xml:space="preserve" id="cmd_fcn_nest50_ade" style="display: none; text-align: left; white-space: pre-wrap">
    python train.py --dataset ADE20K --model fcn  --aux --backbone resnest50
    </code>

    <code xml:space="preserve" id="cmd_enc_nest50_ade" style="display: none; text-align: left; white-space: pre-wrap">
    python train.py --dataset ADE20K --model EncNet --aux --se-loss --backbone resnest50
    </code>

    <code xml:space="preserve" id="cmd_deeplab_resnest50_ade" style="display: none; text-align: left; white-space: pre-wrap">
    python train.py --dataset ADE20K --model deeplab --aux --backbone resnest50
    </code>

    <code xml:space="preserve" id="cmd_deeplab_resnest101_ade" style="display: none; text-align: left; white-space: pre-wrap">
    python train.py --dataset ADE20K --model deeplab --aux --backbone resnest101 --epochs 180
    </code>

    <code xml:space="preserve" id="cmd_deeplab_resnest200_ade" style="display: none; text-align: left; white-space: pre-wrap">
    python train.py --dataset ADE20K --model deeplab --aux --backbone resnest200 --epochs 180
    </code>

    <code xml:space="preserve" id="cmd_deeplab_resnest269_ade" style="display: none; text-align: left; white-space: pre-wrap">
    python train.py --dataset ADE20K --model deeplab --aux --backbone resnest269
    </code>


Pascal Context Dataset
~~~~~~~~~~~~~~~~~~~~~~

==============================================================================  ====================    ====================    =========================================================================================================
Model                                                                           pixAcc                  mIoU                    Command                                                                                      
==============================================================================  ====================    ====================    =========================================================================================================
FCN_ResNeSt50_PContext                                                          79.19%                  51.98%                  :raw-html:`<a href="javascript:toggleblock('cmd_fcn_nest50_pcont')" class="toggleblock">cmd</a>`
DeepLab_ResNeSt50_PContext                                                      80.41%                  53.19%                  :raw-html:`<a href="javascript:toggleblock('cmd_deeplab_nest50_pcont')" class="toggleblock">cmd</a>`
DeepLab_ResNeSt101_PContext                                                     81.91%                  56.49%                  :raw-html:`<a href="javascript:toggleblock('cmd_deeplab_nest101_pcont')" class="toggleblock">cmd</a>`
DeepLab_ResNeSt200_PContext                                                     82.50%                  58.37%                  :raw-html:`<a href="javascript:toggleblock('cmd_deeplab_nest200_pcont')" class="toggleblock">cmd</a>`
DeepLab_ResNeSt269_PContext                                                     83.06%                  58.92%                  :raw-html:`<a href="javascript:toggleblock('cmd_deeplab_nest269_pcont')" class="toggleblock">cmd</a>`
==============================================================================  ====================    ====================    =========================================================================================================

.. raw:: html

    <code xml:space="preserve" id="cmd_fcn_nest50_pcont" style="display: none; text-align: left; white-space: pre-wrap">
    python train.py --dataset pcontext --model fcn --aux --backbone resnest50
    </code>

    <code xml:space="preserve" id="cmd_deeplab_nest50_pcont" style="display: none; text-align: left; white-space: pre-wrap">
    python train.py --dataset pcontext --model deeplab --aux --backbone resnest50
    </code>

    <code xml:space="preserve" id="cmd_deeplab_nest101_pcont" style="display: none; text-align: left; white-space: pre-wrap">
    python train.py --dataset pcontext --model deeplab --aux --backbone resnest101
    </code>

    <code xml:space="preserve" id="cmd_deeplab_nest200_pcont" style="display: none; text-align: left; white-space: pre-wrap">
    python train.py --dataset pcontext --model deeplab --aux --backbone resnest200
    </code>

    <code xml:space="preserve" id="cmd_deeplab_nest269_pcont" style="display: none; text-align: left; white-space: pre-wrap">
    python train.py --dataset pcontext --model deeplab --aux --backbone resnest269
    </code>


ResNet Backbone Models
----------------------

ADE20K Dataset
~~~~~~~~~~~~~~

==============================================================================  ====================    ====================    =============================================================================================
Model                                                                           pixAcc                  mIoU                    Command                                                                                      
==============================================================================  ====================    ====================    =============================================================================================
FCN_ResNet50s_ADE                                                               78.7%                   38.5%                   :raw-html:`<a href="javascript:toggleblock('cmd_fcn50_ade')" class="toggleblock">cmd</a>`
EncNet_ResNet50s_ADE                                                            80.1%                   41.5%                   :raw-html:`<a href="javascript:toggleblock('cmd_enc50_ade')" class="toggleblock">cmd</a>`    
EncNet_ResNet101s_ADE                                                           81.3%                   44.4%                   :raw-html:`<a href="javascript:toggleblock('cmd_enc101_ade')" class="toggleblock">cmd</a>`   
==============================================================================  ====================    ====================    =============================================================================================


.. raw:: html

    <code xml:space="preserve" id="cmd_fcn50_ade" style="display: none; text-align: left; white-space: pre-wrap">
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset ADE20K --model FCN
    </code>

    <code xml:space="preserve" id="cmd_psp50_ade" style="display: none; text-align: left; white-space: pre-wrap">
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset ADE20K --model PSP --aux
    </code>

    <code xml:space="preserve" id="cmd_enc50_ade" style="display: none; text-align: left; white-space: pre-wrap">
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset ADE20K --model EncNet --aux --se-loss
    </code>

    <code xml:space="preserve" id="cmd_enc101_ade" style="display: none; text-align: left; white-space: pre-wrap">
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset ADE20K --model EncNet --aux --se-loss --backbone resnet101
    </code>

Pascal Context Dataset
~~~~~~~~~~~~~~~~~~~~~~

==============================================================================  =====================    =====================    =============================================================================================
Model                                                                           pixAcc                   mIoU                     Command                                                                                      
==============================================================================  =====================    =====================    =============================================================================================
Encnet_ResNet50s_PContext                                                        79.2%                    51.0%                    :raw-html:`<a href="javascript:toggleblock('cmd_enc50_pcont')" class="toggleblock">cmd</a>`  
EncNet_ResNet101s_PContext                                                       80.7%                    54.1%                    :raw-html:`<a href="javascript:toggleblock('cmd_enc101_pcont')" class="toggleblock">cmd</a>` 
==============================================================================  =====================    =====================    =============================================================================================

.. raw:: html

    <code xml:space="preserve" id="cmd_fcn50_pcont" style="display: none; text-align: left; white-space: pre-wrap">
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset PContext --model FCN
    </code>

    <code xml:space="preserve" id="cmd_enc50_pcont" style="display: none; text-align: left; white-space: pre-wrap">
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset PContext --model EncNet --aux --se-loss
    </code>

    <code xml:space="preserve" id="cmd_enc101_pcont" style="display: none; text-align: left; white-space: pre-wrap">
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset PContext --model EncNet --aux --se-loss --backbone resnet101
    </code>


Pascal VOC Dataset
~~~~~~~~~~~~~~~~~~

==============================================================================  ======================    =====================    =============================================================================================
Model                                                                           pixAcc                    mIoU                     Command                                                                                      
==============================================================================  ======================    =====================    =============================================================================================
EncNet_ResNet101s_VOC                                                           N/A                       85.9%                    :raw-html:`<a href="javascript:toggleblock('cmd_enc101_voc')" class="toggleblock">cmd</a>`   
==============================================================================  ======================    =====================    =============================================================================================

.. raw:: html

    <code xml:space="preserve" id="cmd_enc101_voc" style="display: none; text-align: left; white-space: pre-wrap">
    # First finetuning COCO dataset pretrained model on augmented set
    # You can also train from scratch on COCO by yourself
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset Pascal_aug --model-zoo EncNet_Resnet101_COCO --aux --se-loss --lr 0.001 --syncbn --ngpus 4 --checkname res101 --ft
    # Finetuning on original set
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset Pascal_voc --model encnet --aux  --se-loss --backbone resnet101 --lr 0.0001 --syncbn --ngpus 4 --checkname res101 --resume runs/Pascal_aug/encnet/res101/checkpoint.params --ft
    </code>


Test Pretrained
~~~~~~~~~~~~~~~

- Prepare the datasets by runing the scripts in the ``scripts/`` folder, for example preparing ``PASCAL Context`` dataset::

      python scripts/prepare_ade20k.py
  
- The test script is in the ``experiments/segmentation/`` folder. For evaluating the model (using MS),
  for example ``EncNet_ResNet50s_ADE``::

      python test.py --dataset ADE20K --model-zoo EncNet_ResNet50s_ADE --eval
      # pixAcc: 0.801, mIoU: 0.415: 100%|████████████████████████| 250/250


Train Your Own Model
--------------------

- Prepare the datasets by runing the scripts in the ``scripts/`` folder, for example preparing ``ADE20K`` dataset::

    python scripts/prepare_ade20k.py

- The training script is in the ``experiments/segmentation/`` folder, example training command::

    python train.py --dataset ade20k --model encnet --aux --se-loss

- Detail training options, please run ``python train.py -h``. Commands for reproducing pre-trained models can be found in the table.

.. hint::
    The validation metrics during the training only using center-crop is just for monitoring the
    training correctness purpose. For evaluating the pretrained model on validation set using MS,
    please use the command::

        python test.py --dataset pcontext --model encnet --aux --se-loss --resume mycheckpoint --eval


Quick Demo
~~~~~~~~~~

.. code-block:: python

    import torch
    import encoding

    # Get the model
    model = encoding.models.get_model('Encnet_ResNet50s_PContext', pretrained=True).cuda()
    model.eval()

    # Prepare the image
    url = 'https://github.com/zhanghang1989/image-data/blob/master/' + \
          'encoding/segmentation/pcontext/2010_001829_org.jpg?raw=true'
    filename = 'example.jpg'
    img = encoding.utils.load_image(
        encoding.utils.download(url, filename)).cuda().unsqueeze(0)

    # Make prediction
    output = model.evaluate(img)
    predict = torch.max(output, 1)[1].cpu().numpy() + 1

    # Get color pallete for visualization
    mask = encoding.utils.get_mask_pallete(predict, 'pascal_voc')
    mask.save('output.png')


.. image:: https://raw.githubusercontent.com/zhanghang1989/image-data/master/encoding/segmentation/pcontext/2010_001829_org.jpg
   :width: 45%

.. image:: https://raw.githubusercontent.com/zhanghang1989/image-data/master/encoding/segmentation/pcontext/2010_001829.png
   :width: 45%


Citation
--------

.. note::
    * Hang Zhang et al. "ResNeSt: Split-Attention Networks" *arXiv 2020*::

        @article{zhang2020resnest,
        title={ResNeSt: Split-Attention Networks},
        author={Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi and Zhang, Zhi and Lin, Haibin and Sun, Yue and He, Tong and Muller, Jonas and Manmatha, R. and Li, Mu and Smola, Alexander},
        journal={arXiv preprint arXiv:2004.08955},
        year={2020}
        }


    * Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, Amit Agrawal. "Context Encoding for Semantic Segmentation"  *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*::

        @InProceedings{Zhang_2018_CVPR,
        author = {Zhang, Hang and Dana, Kristin and Shi, Jianping and Zhang, Zhongyue and Wang, Xiaogang and Tyagi, Ambrish and Agrawal, Amit},
        title = {Context Encoding for Semantic Segmentation},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2018}
        }
