Context Encoding for Semantic Segmentation (EncNet)
===================================================

Install Package
---------------

- Clone the GitHub repo::
    
    git clone https://github.com/zhanghang1989/PyTorch-Encoding

- Install PyTorch Encoding (if not yet). Please follow the installation guide `Installing PyTorch Encoding <../notes/compile.html>`_.

Test Pre-trained Model
----------------------

.. hint::
    The model names contain the training information. For instance ``FCN_ResNet50_PContext``:
      - ``FCN`` indicate the algorithm is “Fully Convolutional Network for Semantic Segmentation”
      - ``ResNet50`` is the name of backbone network.
      - ``PContext`` means the PASCAL in Context dataset.

    How to get pretrained model, for example ``FCN_ResNet50_PContext``::

        model = encoding.models.get_model('FCN_ResNet50_PContext', pretrained=True)

    The test script is in the ``experiments/segmentation/`` folder. For evaluating the model (using MS),
    for example ``Encnet_ResNet50_PContext``::

        python test.py --dataset PContext --model-zoo Encnet_ResNet50_PContext --eval
        # pixAcc: 0.7888, mIoU: 0.5056: 100%|████████████████████████| 1276/1276 [46:31<00:00,  2.19s/it]

    The command for training the model can be found by clicking ``cmd`` in the table.

.. role:: raw-html(raw)
   :format: html

+----------------------------------+-----------+-----------+-----------+----------------------------------------------------------------------------------------------+------------+
| Model                            | pixAcc    | mIoU      | Note      | Command                                                                                      | Logs       |
+==================================+===========+===========+===========+==============================================================================================+============+
| Encnet_ResNet50_PContext         | 78.9%     | 50.6%     |           | :raw-html:`<a href="javascript:toggleblock('cmd_enc50_pcont')" class="toggleblock">cmd</a>`  | ENC50PC_   |
+----------------------------------+-----------+-----------+-----------+----------------------------------------------------------------------------------------------+------------+
| EncNet_ResNet101_PContext        | 80.3%     | 53.2%     |           | :raw-html:`<a href="javascript:toggleblock('cmd_enc101_pcont')" class="toggleblock">cmd</a>` | ENC101PC_  |
+----------------------------------+-----------+-----------+-----------+----------------------------------------------------------------------------------------------+------------+
| EncNet_ResNet50_ADE              | 79.9%     | 41.2%     |           | :raw-html:`<a href="javascript:toggleblock('cmd_enc50_ade')" class="toggleblock">cmd</a>`    | ENC50ADE_  |
+----------------------------------+-----------+-----------+-----------+----------------------------------------------------------------------------------------------+------------+

.. _ENC50PC: https://github.com/zhanghang1989/image-data/blob/master/encoding/segmentation/logs/encnet_resnet50_pcontext.log?raw=true
.. _ENC101PC: https://github.com/zhanghang1989/image-data/blob/master/encoding/segmentation/logs/encnet_resnet101_pcontext.log?raw=true
.. _ENC50ADE: https://github.com/zhanghang1989/image-data/blob/master/encoding/segmentation/logs/encnet_resnet50_ade.log?raw=true


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

    <code xml:space="preserve" id="cmd_psp50_ade" style="display: none; text-align: left; white-space: pre-wrap">
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset ADE20K --model PSP --aux
    </code>

    <code xml:space="preserve" id="cmd_enc50_ade" style="display: none; text-align: left; white-space: pre-wrap">
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset ADE20K --model EncNet --aux --se-loss
    </code>

Quick Demo
~~~~~~~~~~

.. code-block:: python

    import torch
    import encoding

    # Get the model
    model = encoding.models.get_model('Encnet_ResNet50_PContext', pretrained=True).cuda()
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
    mask = encoding.utils.get_mask_pallete(predict, 'pcontext')
    mask.save('output.png')


.. image:: https://raw.githubusercontent.com/zhanghang1989/image-data/master/encoding/segmentation/pcontext/2010_001829_org.jpg
   :width: 45%

.. image:: https://raw.githubusercontent.com/zhanghang1989/image-data/master/encoding/segmentation/pcontext/2010_001829.png
   :width: 45%

Train Your Own Model
--------------------

- Prepare the datasets by runing the scripts in the ``scripts/`` folder, for example preparing ``PASCAL Context`` dataset::

    python scripts/prepare_pcontext.py

- The training script is in the ``experiments/segmentation/`` folder, example training command::

    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset pcontext --model encnet --aux --se-loss

- Detail training options, please run ``python train.py -h``.

- The validation metrics during the training only using center-crop is just for monitoring the
  training correctness purpose. For evaluating the pretrained model on validation set using MS,
  please use the command::

    CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset pcontext --model encnet --aux --se-loss --resume mycheckpoint --eval

Citation
--------

.. note::
    * Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, Amit Agrawal. "Context Encoding for Semantic Segmentation"  *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*::

        @InProceedings{Zhang_2018_CVPR,
        author = {Zhang, Hang and Dana, Kristin and Shi, Jianping and Zhang, Zhongyue and Wang, Xiaogang and Tyagi, Ambrish and Agrawal, Amit},
        title = {Context Encoding for Semantic Segmentation},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2018}
        }
