#  RFCN for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述

## 简述
R-FCN是一个目标检测网络。位移不变性是卷积网络一个重要的特征，而在检测任务中我们需要网络对物体的位置非常敏感。R-FCN 的提出便是解决分类任务中位移不变性和检测任务中位移可变性直接的矛盾的。同时，针对Faster R-CNN存在的性能瓶颈。在R-FCN中，ROI之后便不存在可学习的参数，从而将相对Faster-RCNN的速度提高了许多。

- 参考实现：

  ```
  url=https://github.com/RebornL/RFCN-pytorch.1.0.git
  commit_id=e32e6db63f13c7f27c42bb3a9c447d42cc0b81e4
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/detection
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |
  | PyTorch 1.11 | numpy==1.21.6 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

    下载PASCAL_VOC2007数据集。
    ```
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/    VOCtrainval_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/    VOCtest_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/    VOCdevkit_08-Jun-2007.tar
    ```


2. 解压数据集。
    ```
    tar xvf VOCtrainval_06-Nov-2007.tar
    tar xvf VOCtest_06-Nov-2007.tar
    tar xvf VOCdevkit_08-Jun-2007.tar
    ```
    在任意目录下新建“data/”目录，将解压后的数据集放置在“data/”目录下。以PASCAL_VOC2007数据集为例，数据集目录结构参考如下所示。

    ```
    data
      VOCdevkit2007 
        ├── annotations_cache
        ├── create_segmentations_from_detections.m
        ├── devkit_doc.pdf
        ├── example_classifier.m
        ├── example_detector.m
        ├── example_layout.m
        ├── example_segmenter.m
        ├── local
        │   ├── VOC2006
        │   │   └── dummy
        │   └── VOC2007
        │       └── dummy
        ├── results
        │   ├── VOC2006
        │   │   └── Main
        │   │       └── dummy
        │   └── VOC2007
        │       ├── Layout
        │       │   └── dummy
        │       ├── Main
        │       │   ├── comp4_det_test_aeroplane.txt
        │       │   ├── comp4_det_test_bicycle.txt
        │       │   ├── ...
        │       │   └── dummy
        │       └── Segmentation
        │           └── dummy
        ├── tree.txt
        ├── viewanno.m
        ├── viewdet.m
        ├── VOC2007
        │   ├── Annotations
        │   │   ├── 000001.xml
        │   │   ├── 000002.xml
        │   │   ├── ...

        │   ├── ImageSets
        │   │   ├── Layout
        │   │   │   ├── test.txt
        │   │   │   ├── train.txt
        │   │   │   ├── trainval.txt
        │   │   │   └── val.txt
        │   │   ├── Main
        │   │   │   ├── aeroplane_test.txt
        │   │   │   ├── aeroplane_train.txt
        │   │   │   ├── aeroplane_trainval.txt
        │   │   │   ├── aeroplane_val.txt
        │   │   │   ├── ...
        │   │   │   └── val.txt
        │   │   └── Segmentation
        │   │       ├── test.txt
        │   │       ├── train.txt
        │   │       ├── trainval.txt
        │   │       └── val.txt
        │   ├── JPEGImages
        │   │   ├── 000001.jpg
        │   │   ├── 000002.jpg
        │   │   ├── ...
        │   ├── SegmentationClass
        │   │   ├── 000032.png
        │   │   ├── 000033.png
        │   │   ├── ...
        │   └── SegmentationObject
        │       ├── 000032.png
        │       ├── 000033.png
        │       ├── ...
        └── VOCcode
            ├── PASemptyobject.m
            ├── PASemptyrecord.m
            ├── PASerrmsg.m
            ├── PASreadrecord.m
            ├── PASreadrectxt.m
            ├── VOCevalcls.m
            ├── VOCevaldet.m
            ├── VOCevallayout.m
            ├── VOCevalseg.m
            ├── VOCinit.m
            ├── VOClabelcolormap.m
            ├── VOCreadrecxml.m
            ├── VOCreadxml.m
            ├── VOCwritexml.m
            └── VOCxml2struct.m           
    ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

## 获取预训练模型

  1. 在data/目录下新建预训练权重放置目录。

      ```
      mkdir pretrained_model
      cd pretrained_model
      ```

  2. 将预训练权重resnet101_rcnn.pth放入当前目录下。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=数据集路径  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=数据集路径  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=数据集路径  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=数据集路径  # 8卡性能
     ```

    --data_path参数填写数据集路径，需写到数据集的一级目录。

    模型训练脚本参数说明如下。  
   ```
   公共参数：
    --dataset                           //数据集路径     
    --epochs                            //重复训练次数
    --bs                                //训练批次大小
    --lr                                //初始学习率
    --amp                               //是否使用混合精度
    --loss-scale                        //混合精度lossscale大小
    --npu_id '0,1,2,3,4,5,6,7'          //单卡训练指定训练用卡
   ```
   
   训练完成后，权重文件默认会写入到和test文件同一目录下，并输出模型训练精度和性能信息到网络脚本test下output文件夹内。


# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 |  FPS | Epochs | AMP_Type | Torch_Version |
| :-----: | :---: | :--: | :----: | :------: | :------: |
| 1p-NPU  | -     | 10.071 | 1      |       O2 |    1.8 |
| 8p-NPU  | 0.7048 | 86.811 | 20    |       O2 |    1.8 |


# 版本说明

## 变更

2023.02.15：更新readme，重新发布。

2020.07.08：首次发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
