# EfficientNet-B5 for PyTorch

-   [概述](#1)
-   [准备训练环境](#2)
-   [开始训练](#3)
-   [训练结果展示](#4)
-   [版本说明](#5)



# 概述

## 简述

EfficientNet是一个新的卷积网络家族，与之前的模型相比，具有更快的训练速度和更好的参数效率。
该模型通过一组固定的缩放系数统一缩放这在网络深度，网络宽度，分辨率这三方面有明显优势。
在EfficientNet中，这些特性是按更有原则的方式扩展的，也就是说，一切都是逐渐增加的。

- 参考实现：

  ```
  url=https://github.com/lukemelas/EfficientNet-PyTorch
  commit_id=7e8b0d312162f335785fb5dcfa1df29a75a1783a
  ```
  
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | pillow==8.4.0 |
  | PyTorch 1.8 | pillow==9.1.0 |
  
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

   用户自行获取原始数据集imagenet2012，将数据集上传到服务器任意路径下并解压。

   数据集目录结构参考如下所示。

   ```
   ├── ImageNet2012
         ├──train
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...   
              ├──...                     
         ├──val  
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...              
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。


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
     bash ./test/train_full_1p.sh --data_path=real_data_path  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=real_data_path  # 单卡性能
     ```
   
   - 单机8卡训练
   
     启动8卡训练。
   
     ```
     bash ./test/train_full_8p.sh --data_path=real_data_path  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=real_data_path  # 8卡性能 
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --data                              //数据集路径
   --arch                              //使用模型，默认：efficientnet-b5
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小
   --lr                                //初始学习率，默认：0.1
   --momentum                          //动量，默认：0.9
   --weight-decay                      //权重衰减，默认：0.0001
   --amp                               //是否使用混合精度
   --loss_scale                        //混合精度losss cale大小
   --pm                                //混合精度类型
   ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1  | FPS | Epochs |  Torch_Version |
| :-----: |:------:|:---:|:------:|:-------------:|
| 1p-NPU  | -      | 79 | 1      |  1.8          |
| 8p-NPU  | 78.595 | 562 | 100    |  1.8          |

> **说明：** 
>单卡训练过程中，混合精度使用**O1**类型；8卡训练过程中，混合精度使用**O2**类型。

# 版本说明

## 变更

2023.02.21：更新readme，重新发布。

## FAQ

无。


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
