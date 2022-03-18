-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)

<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：huawei**

**应用领域（Application Domain）：Object Detection**

**版本（Version）：1.0**

**修改时间（Modified） ：2021.05.20**

_**大小（Size）**_**：318M**

**框架（Framework）：PyTorch1.5**

**模型格式（Model Format）：pth**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于PyTorch框架的FasterRCNN目标检测网络**

<h2 id="概述.md">概述</h2>
Faster R-CNN是截至目前，RCNN系列算法的最杰出产物，two-stage中最为经典的物体检测算法。推理第一阶段先找出图片中待检测物体的anchor矩形框（对背景、待检测物体进行二分类），第二阶段对anchor框内待检测物体进行分类。R-CNN系列物体检测算法的思路都是先产生一些待检测框，再对检测框进行分类。Faster R-CNN使用神经网络生成待检测框，替代了其他R-CNN算法中通过规则等产生候选框的方法，从而实现了端到端训练，并且大幅提速。本文档描述的Faster R-CNN是基于PyTorch实现的版本。    

-   参考实现：

    https://github.com/facebookresearch/detectron2
    
- 适配昇腾 AI 处理器的实现：

  https://gitee.com/ascend/modelzoo/tree/master/built-in/PyTorch/Research/cv/detection/Faster_Mask_RCNN_for_PyTorch

<h2 id="训练环境准备.md">训练环境准备</h2>

硬件环境准备请参见[各硬件产品文档](https://ascend.huawei.com/#/document?tag=developer)。需要在硬件设备上安装固件与驱动。

关键依赖请获取NPU适配版本：

PyTorch

apex

tensor-fused-plugin

另外该代码运行需要从源编译库：

    cd Faster_Mask_RCNN_for_PyTorch
    python3 -m pip install -e ./
## 默认配置

-   训练超参（8卡）：
    -   Batch size: 16(2 per device)
    -   Momentum: 0.9
    -   LR scheduler: step
    -   Learning rate\(LR\): 0.02
    -   Weight decay: 0.0001
    -   Label smoothing: 0.1
    -   Train epoch: 12


## 混合精度训练

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度
在启动脚本中执行训练脚本处配置命令行参数 AMP 1 即可开启NPU上混合精度训练模式。


## 数据集准备

默认使用coco2017数据集，请用户自行获取。数据集路径通过启动脚本的命令行参数--data_path配置。应有如下目录结构

/path/to/dataset/coco

## 快速上手

1.下载预训练模型。

以resnet50为例：

wget https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl

将其置于数据集所在目录下

另附resnet101[下载地址](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl)

2.开始训练。
- 单机单卡

  cd test && bash ./train_full_1p.sh --data_path=/path/to/dataset

- 单机8卡

  cd test && bash ./train_full_8p.sh --data_path=/path/to/dataset
