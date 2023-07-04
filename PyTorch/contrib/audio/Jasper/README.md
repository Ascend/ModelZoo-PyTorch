# Jasper for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

Jasper语音识别网络是基于注意力机制的编码器-解码器架构，如Listen、Attend和Spell(LAS)可以将传统自动语音识别(ASR)系统上的声学、发音和语音模型组件集成到单个神经网络中。在结构上，我们证明了词块模型可以用来代替字素。我们引入了新型的多头注意力架构，它比常用的单头注意力架构有所提升。在优化方面，我们探索了同步训练，定期采样，平滑标签（label smoothing）,也应用了最小误码率优化，这些方法都提升了准确度。我们使用一个单项LSTM编码器进行串流识别并展示了结果。

- 参考实现：

  ```
  url=https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper
  commit_id=0e279a3c7cbfabacbcecb9b5f123d4b532d799f1
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/audio
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

   用户可以自行获取 `LibriSpeech` 原始数据集，并上传到服务器 `/home/dataset` 目录下。也可以直接进入源码包根目录下运行以下两个命令下载数据集并进行预处理。

   ```
   bash scripts/download_librispeech.sh
   bash scripts/preprocess_librispeech.sh
   ```

   数据集目录结构参考如下所示。

   ```
   ├── dataset
        |——LibriSpeech
            ├──dev-clean-wav
            ├──dev-other-wav
            │──librispeech-train-clean-100-wav.json
            │──librispeech-train-clean-360-wav.json      
            ├──librispeech-train-clean-500-wav.json
            │──librispeech-dev-clean-wav.json
            │──librispeech-dev-other-wav.json
            ├──librispeech-test-clean-wav.json                     
            ├──librispeech-test-other-wav.json   
            ├──test-clean-wav
            │──test-other-wav
            │──train-clean-100-wav     
            ├──train-clean-360-wav
            │──train-clean-500-wav
                  
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
     bash ./test/train_full_1p.sh --data_path=/home/dataset/LibriSpeech/  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=/home/dataset/LibriSpeech/ # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/home/dataset/LibriSpeech/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/home/dataset/LibriSpeech/ # 8卡性能
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录，此模型默认为 `/home/dataset/LibriSpeech/`。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --amp                               //是否使用混合精度
   --dataset_dir                       //数据集目录   
   --val_manifests                     //验证集路径
   --model_config                      //模型配置文件  
   --output_dir                        //输出路径
   --lr                                //学习率
   --min_lr                            //最小学习率
   --weight-decay                      //权重衰减
   --prediction-frequency              //在dev set评估之间的steps
   --resume                            //权重路径
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小
   --seed                              //随机种子
   --optimizer                         //优化器
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表


|   NAME   | WER | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |  
| 1p-竞品V |    -    |   10       | 1 |  - | 1.5 |   
| 8p-竞品V |  10.73  |   78      | 30 | - | 1.5 |
| 1p-NPU   |    -    |   4       | 1 | - | 1.5 |
| 8p-NPU   |  10.89  |  34     | 30 | - | 1.5 |


# 版本说明

## 变更

2023.1.10：更新readme，重新发布。


## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md