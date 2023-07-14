# FastSCNN 训练

# the real-time image segmentation FastSCNN

Fast segmentation convolutional neural network (Fast-SCNN), an above real-time semantic segmentation model on high resolution image data (1024x2048px) suits to efficient computation on embedded devices with low memory. Building on existing two-branch methods for fast segmentation, the 'learning to downsample' module  computes low-level features for multiple resolution branches simultaneously. FastSCNN combines spatial detail at high resolution with deep features extracted at lower resolution, yielding an accuracy of 68.0% mean intersection over union at 123.5 frames per second on Cityscapes.

For more detail：https://arxiv.org/abs/1902.04502

## 

## Requirements

use pytorch, you can use pip or conda to install the requirements

```
# for pip
cd $project
pip3.7 install -r requirements.txt
Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
CANN 20210617_5.0.T205
torchvision
```



## 数据集准备

1.从以下网址获取leftImg8bit_trainvaltest.zip作为训练集

https://www.cityscapes-dataset.com/downloads/

2.从以往网址获取gtFine_trainvaltest.zip作为标签

https://www.cityscapes-dataset.com/downloads/

文件结构如下：


```
FastSCNN
|-- configs
|-- datasets
|   |-- cityscapes
|   |   |-- gtFine
|   |   |   |-- test
|   |   |   |-- train
|   |   |   `-- val
|   |   `-- leftImg8bit
|   |       |-- test
|   |       |-- train
|   |       `-- val
|-- docs
|-- test
|-- segmentron
|-- tools

```

将数据集按照以上结构放在代码目录下

## 安装

请注意，本模型使用了新版本的pytorch以及CANN包，具体版本为：20210617_5.0.T205；

![](C:\Users\dilig\Pictures\image-20210824164049265 (2).png)

source 环境变量

```
source ./test/env.sh
```

安装

```
python3 setup.py develop
```

或使用sh脚本安装

```
bash ./test/setup.sh
```



## TRAIN

### 单p训练

source 环境变量

```
source ./test/env.sh
```

运行单p脚本

```
bash ./test/run1p.sh
```



### 多p训练

source 环境变量

```
source ./test/env.sh
```

运行8p脚本

```
bash ./test/run8p.sh
```

模型保存在./runs/checkpoints目录下，以数字命名的pth文件是当前epoch训练得到的权重文件，可用来恢复训练，best_model.pth是当前训练出的最优模型；

运行日志保存至./runs/logs目录下

## TEST

测试精度 

使用sh文件

```
bash test/eval.sh
```

### 精度对比

GPU8p loss scale使用O1 128混合精度获得的结果为：mIoU:64.46

NPU8p loss scale使用O1 128混合精度获得的结果为:   mIoU:63.914

## 注意事项

由于在./FastSCNN/segmentron/modules/csrc/vision.cpp中添加了Licence，可能会导致程序在调用此文件时报错，只需要删除Licence就可以使用


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
