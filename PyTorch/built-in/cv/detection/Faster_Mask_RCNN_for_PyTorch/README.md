# Faster Mask RCNN for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

FasterRCNN是一个业界领先的目标检测网络，他继承了FastRCNN的候选区域+目标识别架构，并在其基础上提出了候选区域网络（RPN）这一概念。通过共享全图卷积特征，FasterRCNN成功做到了让RPN不带来额外时间开销；而RPN的引入则将时下流行的神经网络“注意力”机制引入到了目标检测网络中。这些特性让FasterRCNN在ILSVRC以及COCO 2015等一系列竞赛上收获了第一名的成绩，同时在VGG-16等模型上拥有5fps的高速率。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/detectron2.git
  commit_id=be792b959bca9af0aacfa04799537856c7a92802
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/detection
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动 | [1.0.15](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC1](https://www.hiascend.com/software/cann/commercial?version=5.1.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)|

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始COCO数据集，将数据集上传到服务器任意路径下并解压。

   数据集目录结构参考如下所示。

   ```
    ├── coco2017
    │   ├── annotations
    │          ├── captions_train2017.json
    │          ├── captions_val2017.json
    │          ├── instances_train2017.json
    │          ├── instances_val2017.json
    │          ├── person_keypoints_train2017.json
    │          ├── person_keypoints_val2017.json
    │   ├── train2017
    │          ├── 000000000009.jpg
    │          ├── 000000000025.jpg
    │          ├── ......
    │   ├── val2017
    │          ├── 000000000139.jpg
    │          ├── 000000000285.jpg
    │          ├── ......             
   ```

## 获取预训练模型

模型脚本会自动下载预训练权重文件。若下载失败，请自行准备R-101.pkl权重文件，将文件放到数据集同级路径下。

## 源码编译Detectron2     
编译器版本：gcc & g++ ≥ 5
```
python3.7 -m pip install -e Faster_Mask_RCNN_for_PyTorch

```
>说明：在重装PyTorch之后，通常需要重新编译detectron2。重新编译之前，需要使用`rm -rf build/ **/*.so`删除旧版本的build文件夹及对应的.so文件。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。    
   mask_rcnn启动训练    
   - 单机单卡训练

        启动单卡训练。

        ```
        bash ./test/train_full_1p.sh --data_path=数据集路径  
        ```

   - 单机8卡训练

        启动8卡训练。

        ```
        bash ./test/train_full_8p.sh --data_path=数据集路径  
        ```
    faster_rcnn启动训练     
   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_faster_rcnn_full_1p.sh --data_path=数据集路径  
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_faster_rcnn_full_8p.sh --data_path=数据集路径 
     ```

   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
    AMP                                                                # 开启混合精度
    OPT_LEVEL                                                          # 设置混合精度优化等级为O2
    LOSS_SCALE_VALUE                                                   # 设置损失函数缩放倍率为64
    MODEL.DEVICE                                                       # 指定运行脚本的物理设备
    SOLVER.IMS_PER_BATCH                                               # 指定输入batch中的图片张数
    SOLVER.MAX_ITER                                                    # 指定最大训练迭代数（超过时训练终止）
    MODEL.RPN.NMS_THRESH                                               # 指定NMS阈值
    MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO                           # 指定BOX POOLER采样率
    MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO                          # 指定MASK POOLER采样率
    DATALOADER.NUM_WORKERS                                             # 指定DATALOADER所用进程个数
    SOLVER.BASE_LR                                                     # 指定学习率
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 |  FPS | Epochs | AMP_Type |
| ------- | ----- | ---: | ------ | -------: |
| 1p-NPU1.5 | -     |  10.735 | -      |        O2 |
| 1p-NPU1.8  | -     |  8.23 | -      |       O2 |
| 8p-NPU1.5 | 26.773 | 76.5 | -    |        O2 |
| 8p-NPU1.8  | 32.3 | 58.8 | -    |       O2 |

# 版本说明

## 变更

2022.8.29：更新内容，重新发布。


## 已知问题

无。
