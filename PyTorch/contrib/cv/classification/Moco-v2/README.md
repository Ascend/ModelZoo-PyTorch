# MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
This implements training of MoCo v2 on the Imagenet dataset, mainly modified from [facebookresearch/moco](https://github.com/facebookresearch/moco).

## MoCo v2 Detail
See `moco` directory.

## Requirements
- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
  Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
- Download the ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

## Training
### Unsupervised Training

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8P machine, run:
```
bash test/train_moco_8p.sh --data_path=[path of imagenet]
```
（此步骤需要训练2周左右时间。由于FP16无法使模型收敛，必须使用FP32，并且maxpool有精度问题，需要在cpu完成计算。）

### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8P machine, run: （预训练模型需要训练两周，时间太长，可从此处[下载预训练模型]( https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E8%AE%AD%E7%BB%83/cv/image_classification/MoCoV2/model_moco_epoch_200.pth.tar)）
```
bash test/train_full_8p.sh --data_path=[path of imagenet]
```

### Training Results

| Acc@1 |             Npu\_nums | Epochs   | AMP\_Type | FPS |
| :------: | :------: | :------: | :------: | :------: | 
|    67.272       | 8        | 100      | O1       | 3869 |

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md