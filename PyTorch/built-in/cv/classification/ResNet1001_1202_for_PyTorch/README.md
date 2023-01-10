# ResNet_1001_1202_for_PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述
ResNet是由微软研究院的Kaiming He等四名华人提出，是ImageNet竞赛中分类问题效果较好的网络，
它引入了残差学习的概念，通过增加直连通道来保护信息的完整性，解决信息丢失、梯度消失、梯度爆炸等问题，
让很深的网络也得以训练，可以极快的加速神经网络的训练。ResNet有不同的网络层数，
常用的有18-layer、34-layer、50-layer、101-layer、152-layer。
ResNet18的含义是指网络中有18-layer。


- 参考实现：

  ```
  url=https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/resnet_cifar.py
  commit_id=7779657ec364b5a18bde3817ea5887b289f841f2
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification
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
  | 硬件    | [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install git+https://github.com/NVIDIA/dllogger.git
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，本模型采用的数据集为cifar-10，将数据集上传到服务器任意路径下并解压。数据集目录结构参考如下所示。

   ```
    cifar-10
    └── cifar-10-batches-py
        ├── batches.meta
        ├── data_batch_1
        ├── data_batch_2
        ├── data_batch_3
        ├── data_batch_4
        ├── data_batch_5
        ├── readme.html
        └── test_batch
   ```

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
     bash ./test/train_full_1p.sh --data_path=/data/cifar-10/ --arch=模型名称 --device_id=NPU卡ID 
     bash ./test/train_performance_1p.sh --data_path=/data/cifar-10/ --arch=模型名称 --device_id=NPU卡ID 
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/cifar-10/ --arch=模型名称
     bash ./test/train_performance_8p.sh --data_path=/data/cifar-10/ --arch=模型名称
     ```

   注：arch选resnet1001或resnet1202，默认为resnet1001

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                              //数据集路径
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  ResNet1001训练结果展示表

| NAME     | Acc@1  | FPS     | Epochs | Torch_version |
|--------  | ------ | :------ | ------ | :------------ |
| 1p-竞品A | -      | 106.679 | 1      | -             |
| 8p-竞品A | 94.164 | 124.031 | 200    | -             |
| 1p-NPU   | -      | 150.329  | 1     | 1.8           |
| 8p-NPU   | 93.796 | 124.031  | 200   | 1.8           |

**表 3**  ResNet1202训练结果展示表

| NAME     | Acc@1  | FPS     | Epochs | Torch_version |
|--------  | ------ | :------ | ------ | :------------ |
| 1p-竞品A | -      | 103.328 | -      | -             |
| 8p-竞品A | 92.449 | 108.277 | 200    | -             |
| 1p-NPU   | -      | 112.162  | 1     | 1.8           |
| 8p-NPU   | 92.927 | 94.069   | 200   | 1.8           |

# 版本说明

## 变更

2023.01.09：更新readme发布。

## 已知问题
无。