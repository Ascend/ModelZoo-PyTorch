# SiamMask_for_Pytorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

SiamMask模型是一个实时执行视觉目标跟踪和视频目标分割的框架，实现了视觉目标跟踪和视频目标分割的统一框架，
可以通过级联方式实现多任务模型， 实现在视觉目标跟踪基准上得到实时SOTA结果，
同时在视频目标分割基准上展示出有竞争力的性能，且能高速运行。

- 参考实现：

  ```
  url=https://github.com/foolwood/SiamMask
  commit_id=0eaac33050fdcda81c9a25aa307fffa74c182e36
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/object_tracking
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

  | 配套       | 版本                                                                           |
  |------------------------------------------------------------------------------| ------------------------------------------------------------ |
  | 硬件    | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)  |
  | 固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)                       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  bash make.sh
  ```


## 准备数据集
   
   本模型使用 `Youtube-VOS`, `COCO`, `ImageNet-DET`, `ImageNet-VID` 数据集来训练， 使用`VOT2018` 数据集来测试。
   建议将数据集下载或者软连接至源码包根目录下的`data`目录下，
   否则需要按需修改`SiamMask_for_Pytorch/experiments/SiamMask-base_base/config.json`文件。

1. 获取训练数据集。
   
    依据 `data` 目录下每个子目录中`readme.md`文件中的提示下载所有数据集。

2. 获取测试数据集。
    
    执行以下命令行获取测试数据集。
    ```shell
      cd SiamMask_for_Pytorch/data
      apt install jq
      bash get_VOT2018_data.sh
    ```

## 下载预训练模型

   执行以下命令获取预训练模型。
   ```shell
      cd SiamMask_for_Pytorch/models
      wget http://www.robots.ox.ac.uk/~qwang/resnet.model
  ```

   完成数据集准备以及预训练模型下载后，目录结构如下所示：
   ```
   SiamMask_for_Pytorch
      ├── data
      │   ├── ytb_vos
      │   │   ├── train.json
      │   │   └── crop511
      │   │       └── train
      │   │           ├── 05d77715782
      │   │           └── ...
      │   ├── coco
      │   │   ├── train2017.json
      │   │   └── crop511
      │   │       ├── train2017
      │   │       └── val2017
      │   ├── det
      │   │   ├── train2017.json
      │   │   └── crop511
      │   │       ├── ILSVRS2013_train
      │   │       ├── ILSVRC2014_train_0000          
      │   │       └── ...
      │   ├── vid
      │   │   ├── train.json
      │   │   └── crop511
      │   │       ├── ILSVRC2015_VID_train_0000
      │   │       ├── ILSVRC2015_VID_train_0001          
      │   │       └── ...
      │   ├── VOT2018
      │   │   ├── bag
      │   │   ├── ants3
      │   │   └── ...
      │   └── VOT2018.json
      ├── models
      │   ├── resnet.model
      │   └── ...
      ├── experiments
      ├── datasets
      └── ...           
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
     bash ./test/train_performance_1p.sh    # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_performance_8p.sh    # 8卡性能
     bash ./test/train_full_8p.sh           # 8卡精度
     ```


   模型训练脚本参数说明如下。

   ```
   公共参数：
   --config                            //配置文件路径
   --workers                           //加载数据进程数 
   --batch                             //训练批次大小
   --epoch                             //重复训练次数
   ```

   **默认情况下会将checkpoint保存在output文件夹下，如果想从头训练，请移除output文件夹再开始训练**

# 训练结果展示

**表 2**  训练结果展示表

|  NAME  | SiamMask_Loss | FPS  | Npu_nums | Epochs | AMP_Type | CPU  |
|:------:|:-------------:|:----:|:--------:|:------:|:--------:|:----:|
| 1p-竞品V |       -       | 193  |    1     |   1    |    -     | x86  |
| 8p-竞品V |       -       | 846  |    8     |   1    |    -     | x86  |
| 8p-竞品V |    2.5939     | 812  |    8     |   20   |    -     | x86  |
| 1p-NPU |       -       | 236  |    1     |   1    |    O1    | ARM  |
| 8p-NPU |       -       | 1724 |    8     |   1    |    O1    | ARM  |
| 8p-NPU |    2.5940     | 1615 |    8     |   20   |    O1    | ARM  |

# 版本说明

## 变更

2023.01.10：Readme整改发布。

## 已知问题

Q:为什么要在配置文件`$SiamMask-base/experiments/siammask_base/config.json`中修改学习率以及权重?

A:详见 [论文](https://arxiv.org/abs/1812.05050) 的 `3.3 Implementation details` 部分。