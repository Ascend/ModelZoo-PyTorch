# Transformer for PyTorch

- [概述](概述.md)

- [准备训练环境](准备训练环境.md)

- [开始训练](开始训练.md)

- [训练结果展示](训练结果展示.md)

- [版本说明](版本说明.md)


# 概述

## 简述

Transformer模型通过跟踪序列数据中的关系来学习上下文并因此学习含义。该模型使用全Attention的结构代替了LSTM，抛弃了之前传统的Encoder-Decoder模型必须结合CNN或者RNN的固有模式，在减少计算量和提高并行效率的同时还取得了更好的结果。

- 参考实现：

  ```
  url=https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer
  commit_id=be349d90738e543b4106a5492b8573fad2b72c24
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
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |


- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  在模型源码包根目录下执行命令。
  ```
  pip install -r requirements.txt
  ```

## 准备数据集

1. 获取数据集。

   用户自行下载 `WMT` (Workshop on Machine Translation)数据集，并将数据集上传到源码包中的 `./examples/translation` 目录下并解压。

   **表 2**  数据集简介表

   | 来源  | 名称                                                         |
   | :---- |:----------------------------------------------- |
   | wmt13 | training-parallel-europarl-v7.tgz |
   | wmt13 | training-parallel-commoncrawl.tgz |
   | wmt17 | training-parallel-nc-v12.tgz |
   | wmt17 | dev.tgz |
   | wmt14 | test-full.tgz       |

   其中，前四项语料为训练集+验证集；最后一项语料为测试集。


2. 数据预处理。

   进入源码包根目录下执行下面脚本。
   ```
   sh run_preprocessing.sh
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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/ # 单卡性能
     ```
   
   - 单机8卡训练
   
     启动8卡训练。
   
     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/ # 8卡性能
     ```
   

   --data\_path参数填写 `run_preprocessing.sh` 中 `DATASET_DIR` 的路径。

   模型训练脚本参数说明如下。
   ```
   公共参数：
   --data_path                         //数据集路径
   --addr                              //主机地址
   --arch                              //使用模型，默认：transformer_wmt_en_de
   --optimizer                         //优化器
   --max_epoch                         //重复训练次数
   --max-sentences                     //训练批次大小
   --lr                                //初始学习率，默认：0.0006
   --weight_decay                      //权重衰减，默认：0.0
   --amp                               //是否使用混合精度
   --amp-level                         //混合精度等级
   --device-id                         //训练设备卡号   
   ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 3**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V |   -   | -  |   1    |    -     |      1.5      |
| 8p-竞品V | - | - |  3   |    -     |      1.5      |
|  1p-NPU  |   -   | 869.903  |   1    |    O2    |      1.8      |
|  8p-NPU  |  10.8858  | 6326.66  |  3   |    O2    |      1.8      |

# 版本说明

## 变更

2022.11.11：更新torch1.8版本，重新发布。

2021.01.12：首次发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md