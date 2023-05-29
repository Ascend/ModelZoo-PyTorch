# DAL for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

DAL模型是一个高效的目标检测模型，它提出了匹配度度量来评估动态锚点的定位潜力，以此来更有效地分配标签。

- 参考实现：

  ```
  url=https://github.com/ming71/DAL
  commit_id=48cd29fdbf5eeea1b5b642bd1f04bbf1863b31e3
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

- 安装依赖库。
  ```
  apt-get install libgl1 libgeos-dev
  ```
  
- 安装torch-warmup-lr。

  ```
  # 处理ca证书不通过问题
  apt-get install ca-certificates
  git config --global http.sslverify false
  # 安装torch-warmup-lr
  git clone https://github.com/lehduong/torch-warmup-lr.git
  cd torch-warmup-lr
  python3 setup.py install
  cd ..
  ```


## 准备数据集

1. 获取数据集。

   请用户自行获取原始数据集**UCAS-AOD**，将数据集上传到服务器任意路径下并解压。
   数据集展开后参考结构如下所示：

   ```
   UCAS_AOD
    └───AllImages
    │   │   P0001.png
    │   │   P0002.png
    │   │	...
    │   └───P1510.png
    └───Annotations
    │   │   P0001.txt
    │   │   P0002.txt
    │   │	...
    │   └───P1510.txt       
    └───ImageSets 
    │   │   train.txt
    │   │   val.txt
    │   └───test.txt  
    └───Test
    │   │   P0003.png
    │   │	...
    │   └───P1508.txt 
    └───CAR
    └───PLANE
    └───Neg            
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

     ```shell
     bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```shell
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  #8卡性能
     ```
   
   - 获得推理结果

     推理启动脚本。
      ```shell 
      bash ./eval.sh
      ```

   注意：训练过程中模型会自动下载预训练模型，若预训练模型下载失败，可以查看训练日志，获取下载地址和存放路径，手动下载后放到对应的位置。
   
   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --opt_level                         //混合精度类型，默认为O1
   --dataset                           //数据集，默认为UCAS_AOD
   --npus_per_node                     //每个节点上npu设备数目    
   --learning-rate                     //初始学习率
   --batch_size                        //训练批次大小
   --epochs                            //重复训练次数
   --weight                            //模型权重
   --MASTER_PORT                       //主机端口号
   --manual_seed                       //随机数种子设置
   ```


# 训练结果展示

**表 2**  训练结果展示表

|   NAME   |  mAP   |  FPS   | Epochs | Batch_Size | Torch_Version |
| :------: | :----: | :----: | :----: | :--------: | :-----------: |
| 1p-竞品V | 0.8987 |  7.3   |  100   |     2      |      1.5      |
| 1p-竞品V | 0.9076 |  9.52  |  100   |     8      |      1.5      |
| 8p-竞品V | 0.9089 | 17.78  |  100   |     64     |      1.5      |
|  1p-NPU  |  0.91  | 10.802 |  100   |     8      |      1.5      |
|  8p-NPU  | 0.916  | 63.880 |  100   |     64     |      1.5      |

# 版本说明

## 变更

2023.02.21：更新readme，重新发布。

## FAQ

无。