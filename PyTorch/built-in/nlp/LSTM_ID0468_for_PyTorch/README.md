# LSTM for PyTorch

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


- 安装kaldi(可选，首次处理TIMIT原始数据集需安装)。

   ```
   chmod +x install_kaldi.sh
   ./install_kaldi.sh

   注意：install_kaldi.sh 根据所使用linux环境做适当修改。例如 centos 环境，将脚本中apt修改为yum;make -j 32, 数字32也可根据机器硬件条件相应修改。
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

| Type | Acc@1 | FPS       | Epochs   | AMP_Type | Torch_Version |
| :------: | :------:  | :------: | :------: | :------: | :------: |
| NPU-1p | 80.5 | 15.225 | 1      | O2    |       |
| NPU-8p | - | - | 30    | O2  |     |

# 版本说明

## 变更

2023.03.03：更新readme，重新发布。

2021.07.08：首次发布。

## FAQ

无。