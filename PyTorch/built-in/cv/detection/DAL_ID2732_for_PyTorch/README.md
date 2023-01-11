# DAL_for_PyTorch

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
  | 硬件 | [1.0.13](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [21.0.4](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.0.4](https://www.hiascend.com/software/cann/commercial?version=5.0.4) |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  安装依赖库
  ```
  # 安装依赖库
  apt-get install libgl1 libgeos-dev
  # 安装requirements.txt
  pip3 install -r requirements.txt
  ```
  
  安装torch-warmup-lr

  ```
  # 处理ca证书不通过问题
  apt-get install ca-certificates
  git config --global http.sslverify false
  # 安装torch-warmup-lr
  git clone https://github.com/lehduong/torch-warmup-lr.git
  cd torch-warmup-lr
  python3.7 setup.py install
  cd ..
  ``` 


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集[UCAS-AOD](https://hyper.ai/datasets/5419)，将数据集上传到服务器任意路径下并解压。
   数据集展开后结构如下所示：

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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/    
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/   
     ```
    
   - 获得推理结果
      ```shell 
      # 推理启动脚本：eval.sh
      bash ./eval.sh
      ```

   --data_path参数填写数据集路径。第一次训练如果预训练模型下载失败，可以查看下载日志，获取下载地址和存放路径，手动下载后放到对应位置

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --opt_level                         //amp类型，默认为O1
   --dataset                           //数据集，默认为UCAS_AOD
   --npus_per_node                     //每个节点上npu设备数目    
   --learning-rate                     //初始学习率
   --batch-size                        //训练批次大小
   --epoch                             //重复训练次数
   --warm_up_epoch                     //warm up
   ```


# 训练结果展示

**表 2**  训练结果展示表

| NAME     | mAP    | FPS      | Epochs | batch  | Torch_version |
|--------  | ------ | :------  | ------ | ------ | :------------ |
| 1p-竞品V | 0.8987 | 7.3      | 100    | 2      | -             |
| 1p-竞品V | 0.9076 | 9.52     | 100    | 8      | -             |
| 8p-竞品V | 0.9089 | 17.78    | 100    | 64     | -             |
| 1p-NPU   | 0.91   | 10.802   | 100    | 8      | 1.5           |
| 8p-NPU   | 0.916  | 63.880   | 100    | 64     | 1.5           |

# 版本说明

## 变更

2023.01.10：更新Readme发布。

## 已知问题

无。