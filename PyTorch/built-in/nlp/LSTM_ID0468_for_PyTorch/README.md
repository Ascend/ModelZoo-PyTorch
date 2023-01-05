# LSTM_ID0468_for_PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

长短期记忆（Long Short Term Memory，LSTM）是RNN的一种，最早由Hochreiter和Schmidhuber（1977）年提出，该模型克服了一下RNN的不足，通过刻意的设计来避免长期依赖的问题。现在很多大公司的翻译和语音识别技术核心都以LSTM为主。LSTM+CTC神经网络就是声学特征转换成音素这个阶段，该阶段的模型被称为声学模型。

- 参考实现：

  ```
  url=https://github.com/Diamondfan/CTC_pytorch/tree/master/timit
  commit_id=eddd2550224dfe5ac28b6c4d20df5dde7eaf6119
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/nlp
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
  | 硬件       | [1.0.11](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [21.0.2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.0.2](https://www.hiascend.com/software/cann/commercial?version=5.0.2) |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```
  1. 安装kaldi(可选，首次处理TIMIT原始数据集需安装)

      chmod +x install_kaldi.sh
      ./install_kaldi.sh

  注意：install_kaldi.sh 根据所使用linux环境做适当修改。例如 centos 环境，将脚本中apt修改为yum;make -j 32, 数字32也可根据机器硬件条件相应修改
       请确认服务器环境网络通畅，否则会导致安装失败
  
  2. 安装依赖    
      pip3.7 install -r requirements.txt
  ```

## 准备数据集

1. 请用户自行获取TIMIT语音数据集并放置服务器的任意目录下。
2. 数据集目录结构如下所示。
    ```
    |--TIMIT
        |--DOC
        |--TEST
        |--TRAIN
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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/
     ```

   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --amp                               //是否使用混合精度
   --addr                              //主机地址
   --dist_backend                      //通信后端
   --dist_url                          //设置分布式训练网址
   --multiprocessing_distributed       //是否使用多卡训练
   --world_size                        //分布式训练节点数量
   --device_list                       //多卡训练指定训练用卡
   --opt_level                         //混合精度类型
   ctc_config.yaml中配置的参数：
   num_epoches                         //重复训练次数
   batch_size                          //训练批次大小
   lr_decay                            //学习率，默认：0.5
   weight_decay                        //权重衰减，默认：0.0001
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| Type | Acc@1 | FPS       | Epochs   |
| :------: | :------:  | :------: | :------: |
| NPU-1p | 80.5 | 15.225 | 1      |
| NPU-8p | - | - | 30    |

# 版本说明

## 变更

2023.01.03：更新readme，重新发布。

2021.07.08：首次发布。

## 已知问题

无。