# RCF for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述

## 简述

边缘检测是计算机视觉中的一个基本问题。近年来，卷积神经网络(CNNs)在这一领域取得了显著的进展。现有的方法采用特定层次的深度cnn，由于尺度和纵横比的变化，可能无法捕捉到复杂的数据结构。RCF将所有卷积特性封装成更具判别性的表示形式，这很好地利用了丰富的特性层次结构，并且可以通过反向传播进行训练。RCF充分利用了目标的多尺度、多层次信息，全面地进行图像对图像的预测。使用VGG16网络，在几个可用的数据集上实现了最先进的性能。

- 参考实现：

  ```
  url=https://github.com/mayorx/rcf-edge-detection
  commit_id=68341dfcadd517db8cdf502f9740b7330496cbfa
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/detection
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.2.2.post3；pillow==8.4.0 |
  | PyTorch 1.8 | torchvision==0.9.1；pillow==9.1.0 |
  
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


## 准备数据集

1. 获取数据集。

   请用户自行准备好数据集，包含训练集和验证集两部分，可选用的数据集包括BSDS500、PASCAL等。将准备好的数据集上传至服务器任意文目录下并解压，解压后训练集图片分别位于“HED-BSDS/train/”和“PASCAL/aug_data/”文件夹路径下，该目录下每个文件夹代表一个尺度和是否有翻转，且同一文件夹下的所有图片都有相同的标签。当前提供的训练脚本中，是以HED-BSDS数据集为例。在使用其他数据集时，修改数据集路径。数据集目录结构参考：

   ```
   ├─data
       ├──HED-BSDS_PASCAL
            ├──gt
            ├──BSR
            ├──HED-BSDS
                  ├─────test 
                       ├──图片1、2、3、4、…、200
                  ├─────train
                       ├──aug_data       
                       ├──aug_data_scale_0.5
                       ├──aug_data_scale_1.5
                       ├──aug_gt
                       ├──aug_gt_scale_0.5
                       ├──aug_gt_scale_1.5
            ├──PASCAL  
                 ├──aug_data
                 ├──aug_gt
                 ├──train_pair.lst
            ├──bsds_pascal_train_pair.lst                                   
   ```
   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。 


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
   --data_path                         //数据集路径
   --batch_size                        //训练批次大小
   --resume                            //加载ckpt文件
   --workers                           //加载的线程数
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Acc@1 |   FPS   | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :-----: | :----: | :------: | :-----------: |
|  1p-NPU  |   -   | 38.217  |   1    |    O2    |      1.8      |
|  8p-NPU  | 79.1  | 117.160 |   30   |    O2    |      1.8      |


# 版本说明

## 变更

2023.02.28：更新readme，重新发布。

2022.02.14：首次发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md