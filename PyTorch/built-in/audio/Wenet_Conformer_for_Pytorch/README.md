# Wenet Conformer for PyTorch

- [概述](#概述)
- [准备训练环境](#准备训练环境)
- [开始训练](#开始训练)
- [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)

# 概述
Wenet是一款开源的、面向工业落地应用的语音识别工具包，主要特点是小而精，它不仅采用了现阶段最先进的网络设计Conformer，还用到了U2结构实现流式与非流式框架的统一。



- 参考实现：

  ```
  url=https://github.com/wenet-e2e/wenet.git
  commit_id=abcc9c57acbb534ae761b0fc61ef86c7610d2d94
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/audio
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version |   三方库依赖版本    |
  | :-----------: | :-----------------: |
  | PyTorch 1.11  | torch_audio==0.11.0 |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  在模型源码包根目录下执行命令。

  ```
  pip3 install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行下载 `aishell-1` 数据集，并将下载好的数据集放置服务器的任意目录下。该数据集包含由 400 位说话人录制的超过 170 小时的语音。数据集目录结构参考如下所示。

   ```
    aishell-1
       ├── data_aishell.tgz
       |
       └── resource_aishell.tgz
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

   该模型支持单机8卡训练。
   - 单机8卡训练

     启动8卡训练。

     ```
     cd examples/aishell/s0/test
     bash train_full_8p.sh --stage=起始stage --stop_stage=终止stage --data_path=/data/xxx/  # 8卡精度
     bash train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

   模型训练脚本参数说明如下。

   ```shell
   --stage              //模型训练的起始阶段，默认为-1，即从数据下载开始启动训练。若之前数据下载、准备、特征生成等阶段已完成，可配置--stage=4开始训练。
   --stop_stage         //模型训练的终止阶段
   --data_path          //数据集路径
   ```

   > **说明：**
   > 
   > --stage <-1 ~ 5>、--stop_stage <-1 ~ 5>：控制模型训练的起始、终止阶段。模型包含 -1 ~ 5 训练阶段，其中 -1 ~ 3 为数据下载、准备、特征生成等阶段，4为模型训练，5为ASR任务评估。首次运行时请从 -1 开始，-1 ~ 3 阶段执行过一次之后，后续可以从stage 4 开始训练。
   > 
   > --data_path参数填写数据集路径，需要写到数据集的一级目录。

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Error | FPS(iters/sec) | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :------------: | :----: | :------: | :-----------: |
| 8p-竞品A |   -   |      800.44        |   -    |    -     |      1.11      |
| 8p-NPU  |   -   |      500.28        |   -    |    -     |      1.11      |

**表 3** conformer result
* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.002, batch size 18, 4 gpu, acc_grad 4, 240 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 20

| decoding mode             | WER   |
|:------:|:------:|
| ctc greedy search        | 4.81  |

# 版本说明

## 变更

2023.06.07：首次发布。

## FAQ


## 代码涉及公网地址

代码涉及公网地址参考[public_address_statement.md](./public_address_statement.md)