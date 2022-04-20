**DETR**: End-to-End Object Detection with Transformers
========
PyTorch training code and pretrained models for **DETR** (**DE**tection **TR**ansformer).
We replace the full complex hand-crafted object detection pipeline with a Transformer, and match Faster R-CNN with a ResNet-50, obtaining **42 AP** on COCO using half the computation power (FLOPs) and the same number of parameters. Inference in 50 lines of PyTorch.

# Usage - Object detection
There are no extra compiled components in DETR and package dependencies are minimal,
so the code is very simple to use. We provide instructions how to install dependencies via conda.
First, clone the repository locally:
```
git clone -b DETR https://gitee.com/eason-hw/ModelZoo-PyTorch.git
```

Install pycocotools (for evaluation on COCO) and scipy (for training):
```
pip3 install -r requirements.txt
```
That's it, should be good to train and evaluate detection models.

## Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
opt/npu/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## Training
To train baseline DETR on a single node with 8 gpus for 300 epochs run:

```
# env
cd ModelZoo-PyTorch/PyTorch/contrib/cv/detection/DETR
dos2unix ./test/*.sh

# training 1p performance
bash test/train_performance_1p.sh   --data_path=YourDataPath

# training 8p performance
bash test/train_performance_8p.sh  --data_path=YourDataPat

# training 8p accuracy
bash test/train_full_8p.sh  --data_path=YourDataPath
```

We train DETR with AdamW setting learning rate in the transformer to 1e-4 and 1e-5 in the backbone.
Horizontal flips, scales and crops are used for augmentation.
Images are rescaled to have min size 800 and max size 1333.
The transformer is trained with dropout of 0.1, and the whole model is trained with grad clip of 0.1.


## Performance (opt_level = O0)
```
DEVICE   |   Epochs/steps   |  FPS    |  LOSS
GPU(1P)  |     1000 steps   |  12.7   |  
NPU(1P)  |     1000 steps   |   0.2   |
GPU(8P)  |      2 Epochs    |  73.5   | 24.4104
NPU(8P)  |      2 Epochs    | 0.489   | 24.9557
```
