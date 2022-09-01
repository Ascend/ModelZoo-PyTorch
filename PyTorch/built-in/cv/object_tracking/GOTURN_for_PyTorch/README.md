# GOTURN for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

GOTURN是Generic Object Tracking Using Regression Networks的缩写，是一种基于深度学习的跟踪算法。大多数跟踪算法都以在线方式进行训练。换句话说，跟踪算法学习在运行时不停获取被跟踪对象的特点。因此，许多实时跟踪器依赖于在线学习算法，这些算法通常比基于深度学习的解决方案快得多。GOTURN通过以离线方式学习对象的运动，改变了我们将深度学习应用于跟踪问题的方式。GOTURN模型在数千个视频序列上进行训练，不需要在运行时进行任何学习。

- 参考实现：

  ```
  url=https://github.com/nrupatunga/goturn-pytorch
  commit_id=bb3b3b418ac361aa5782b0a7569cc2191b1c30fc
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/detection/
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

  | 配套        | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动   | [22.0.RC3](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.1.RC1](https://www.hiascend.com/software/cann/commercial?version=6.1.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)|

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖（根据模型需求，按需添加所需依赖）。

  ```
  pip3 install -r requirements.txt
  ```
- 配置运行环境

  ```
  # 配置适配NPU的pytorch-lightning
  bash pytorch_lightning_tonpu.sh
  # 配置依赖文件路径
  cd ../
  source settings.sh
  ```

## 准备数据集

1. 获取数据集。

   返回项目目录，和src文件夹同级的目录
   ```
   cd scritpts
   bash src/scritpts/download_data.sh
   
   # ILSVRC2014_Det数据集
   mkdir ILSVRC2014_Det
   # ILSVRC2014_DET_train.tar 解压后，需要进一步解压：
   bash src/scripts/unzip_imagenet.sh ./ILSVRC2014_DET_train ./ILSVRC2014_Det/images/
   # ILSVRC2014_DET_bbox_train.tgz 解压后，放到ILSVRC2014_Det/gt中
  
   # ALOV数据集
   mkdir ALOV
   # alov300++_frames.zip 解压后，放到ALOV/images/中
   # alov300++GT_txtFiles.zip 解压后，放到ALOV/gt/中  
   ```

   准备好数据集后放到 ./dataset 目录如下

   ```
   ├── dataset
         ├── ILSVRC2014_Det（约85G）
             ├── images
                 ├── n00007846
                     ├── n00007846_103856.JPEG 
                     ├── n00007846_104163.JPEG
                     ......
                 ├── n00141669
                     ├── n00141669_119.JPEG
                     ├── n00141669_119.JPEG
                     ......
                 ......
             ├── gt
                 ├── n00007846
                     ├── n00007846_103856.xml
                     ├── n00007846_104163.xml
                     ......   
                 ......
         ├── ALOV（约11G）
             ├── images
                ├── 01-Light
                    ├── 01-Light_video00001
                        ├── 00000001.jpg
                        ├── 00000002.jpg
                        ......
                ├── 02-SurfaceCover
                    ├── 02-SurfaceCover_video00001
                        ├── 00000001.jpg
                        ├── 00000002.jpg
                        ......
                ......
             ├── gt
                ├── 01-Light
                    ├── 01-Light_video00001.ann
                    ├── 01-Light_video00002.ann
                    ......
                ├── 02-SurfaceCover
                    ├── 02-SurfaceCover_video00001.ann
                    ├── 02-SurfaceCover_video00002.ann
                    ......
                ......
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理
    - 本模型不涉及

## 获取预训练模型（可选）

- 从https://github.com/nrupatunga/goturn-pytorch/tree/master/src/goturn/models/pretrained 中获取预训练模型放到src/goturn/models/pretrained中

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
     bash ./test/train_full_1p.sh --data_path=real_path
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh  --data_path=real_path
     ```

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --config                            //训练配置
   --imagenet_path ../data/ILSVRC2014_Det/  //ILSVRC2014数据集路径
   --alov_path ../data/ALOV/  //ALOV数据集路径
   --save_path ../caffenet/  //训练模型保存路径
   --epochs 20  //迭代的最大次数
   --npus 8  //使用的
   --batch_size 3  
   --pretrained_model ../goturn/models/pretrained/caffenet_weights.npy  //预训练模型
   ```


# 训练结果展示

**表 2**  训练结果展示表

| NAME    | val-loss | 性能(it/s) | Epoch | AMP_Type |
|---------|----------|---------:|-------|---------:|
| 1p-竞品V  | -        |      1.1 | 1     |       O2 |
| 1p-NPU  | -        |     8.58 | 1     |       O2 |
| 8p-竞品V  | 64.3770  |     8.64 | 20    |       O2 |
| 8p-NPU  | 64.3497  |    28.32 | 20    |       O2 |


备注：一定要有竞品和NPU。

# 版本说明

## 变更

2022.08.31：首次发布

## 已知问题

**_当前发行版本中存在的问题描述。_**

无。











