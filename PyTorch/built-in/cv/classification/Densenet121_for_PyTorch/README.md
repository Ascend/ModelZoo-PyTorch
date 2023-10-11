# DenseNet121 for PyTorch

-   [概述](#1)
-   [准备训练环境](#2)
-   [开始训练](#3)
-   [训练结果展示](#4)
-   [版本说明](#5)



# 概述

## 简述

DenseNet-121是一个经典的图像分类网络，对于一个L层的网络，DenseNet共包含L\*（L+1）/2个连接，相比ResNet，这是一种密集连接，他的名称也由此而来，另一大特色为通过特征在channel上的连接来实现特征重用（feature reuse），这些特点让DenseNet在参数和计算成本更少的情形下实现比ResNet更优的性能。

- 参考实现：

  ```
  url=https://github.com/pytorch/examples/tree/main/imagenet
  commit_id=f5bb60f8e6b2881be3a2ea8c9a3d43e676aa2340
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
  | PyTorch 1.5 | torchvision==0.2.2.post3；pillow==8.4.0 |
  | PyTorch 1.8 | torchvision==0.9.1；pillow==9.1.0 |
  | PyTorch 1.11 | torchvision==0.12.0 |
  | PyTorch 2.1 | torchvision==0.16.0 |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  
  ```
  pip install -r 1.5_requirements.txt  # Pytorch1.5版本

  pip install -r 1.8_requirements.txt  # Pytorch1.8版本

  pip install -r 1.11_requirements.txt  # Pytorch1.11版本

  pip install -r 2.1_requirements.txt  # Pytorch2.1版本
  ```
  > **说明:**
  > 只需执行一条对应的PyTorch版本依赖安装命令。


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集imagenet2012，将数据集上传到服务器任意路径并解压。

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
   --addr                              //主机地址
   --arch                              //使用模型，默认：densenet121
   --workers                           //加载数据进程数
   --epoch                             //重复训练次数
   --batch-size                        //训练批次大小
   --lr                                //初始学习率，默认：0.1
   --momentum                          //动量，默认：0.9
   --weight_decay                      //权重衰减，默认：0.0001
   --amp                               //是否使用混合精度
   --loss-scale                        //混合精度lossscale大小
   --opt-level                         //混合精度类型
   多卡训练参数：
   --multiprocessing-distributed       //是否使用多卡训练
   --device-list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
   ```


# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
|  1p-NPU-ARM  |   -   | 1038  |   1    |    O2    |      1.8      |
|  8p-NPU-ARM  | 74.658 | 7445  |  90   |    O2    |      1.8      |
|  1p-NPU-非ARM  |   -   | 1180.78  |   1    |    O2    |      1.8      |
|  8p-NPU-非ARM  | - | 7315.34  |  90   |    O2    |      1.8      |


# 版本说明

## 变更

2023.04.24：更新readme，重新发布。

## FAQ

无。


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
