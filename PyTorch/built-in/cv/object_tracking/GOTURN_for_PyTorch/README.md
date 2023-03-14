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
  code_path=PyTorch/built-in/cv/object_tracking/
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.2.2.post3 |
  | PyTorch 1.8 | torchvision==0.9.1 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。

- 配置运行环境。

  ```
  # 配置适配NPU的pytorch-lightning
  cd ${模型文件夹名称}/src/scripts/
  bash pytorch_lightning_tonpu.sh
  # 配置依赖文件路径
  cd ../
  source settings.sh
  ```

## 准备数据集

1. 获取数据集。

   请用户参考以下方法进行数据集获取，在源码包根目录下新建文件夹`ILSVRC2014_Det/` 和 `ALOV/`，并在两个目录下分别建立两个二级目录，将数据集存放在对应目录下，操作方法如下所示。
   ```
   # 在源码包根目下执行命令，下载数据集
   bash ./src/scritpts/download_data.sh
   
   mkdir ILSVRC2014_Det ALOV
   cd ILSVRC2014_Det
   mkdir images 
   mkdir gt  # ILSVRC2014_DET_bbox_train.tgz 解压后，放到ILSVRC2014_Det/gt中
   cd ../
   tar -xvf ILSVRC2014_DET_train.tar 解压后，需要通过unzip_imagenet.sh脚本进一步解压
   bash ./src/scripts/unzip_imagenet.sh ./ILSVRC2014_DET_train ./ILSVRC2014_Det/images/
   
   cd ALOV
   mkdir images  # alov300++_frames.zip 解压后，放到ALOV/images/中
   mkdir gt  # alov300++GT_txtFiles.zip 解压后，放到ALOV/gt/中
   ```

   将准备好数据集放到源码包根目录下新建的` dataset/` 目录下，数据集目录结构参考如下所示。

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


## 获取预训练模型

- 请用户根据“参考实现”源码链接，将源码 `goturn-pytorch/src/goturn/models/pretrained/` 目录下的 caffenet_weights.npy 文件下载到本模型新建的 `./src/goturn/models/pretrained/`  目录下。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd ${模型文件夹名称}/
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --epochs                            //训练周期数
   --batch_size                        //训练批次大小
   --momentum                          //动量
   --wd                                //权重衰减
   --lr                                //初始学习率
   --max_steps                         //迭代次数
   --seed                              //随机数种子设置
   --device                            //训练卡ID设置
   --pretrained_model                  //加载预训练模型
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

| NAME    | val-loss | 性能(it/s) | Epochs | AMP_Type |
|:-------:|:-------:|:---------:|:-------:|:--------:|
| 1p-竞品V  | -        |      1.1 | 1     |       O2 |
| 8p-竞品V  | 64.3770  |     8.64 | 20    |       O2 |
| 1p-NPU   | -        |     8.58 | 1     |       O2 |
| 8p-NPU   | 64.3497  |    28.32 | 20    |       O2 |


# 版本说明

## 变更

2023.03.14：更新readme，重新发布。

2022.08.31：首次发布

## FAQ

无。