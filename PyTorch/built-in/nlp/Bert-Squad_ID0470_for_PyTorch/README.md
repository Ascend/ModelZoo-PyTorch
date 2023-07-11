# Bert-Squad for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述
BERT的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的Encoder，是一种用于自然语言处理（NLP）的预训练技术。

BERT-base模型是一个12层，768维，12个自注意头（self attention head），110M参数的神经网络结构，它的整体框架是由多层transformer的encoder堆叠而成的。每一层的encoder则是由一层muti-head-attention和一层feed-forward组成，每个attention的主要作用是通过目标词与句子中的所有词汇的相关度，对目标词重新编码。所以每个attention的计算包括三个步骤：计算词之间的相关度，对相关度归一化，通过相关度和所有词的编码进行加权求和获取目标词的编码，本文档描述的是BERT-base模型在PyTorch上实现的版本。

BERT-Large模型是一个24层，1024维，24个自注意头（self attention head），110M参数的神经网络结构，它的整体框架是由多层transformer的encoder堆叠而成的。每一层的encoder则是由一层muti-head-attention和一层feed-forward组成，每个attention的主要作用是通过目标词与句子中的所有词汇的相关度，对目标词重新编码。所以每个attention的计算包括三个步骤：计算词之间的相关度，对相关度归一化，通过相关度和所有词的编码进行加权求和获取目标词的编码，本文档描述的是BERT-base模型在PyTorch上实现的版本。

- 参考实现：

  ```
  url=https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT
  commit_id=499fb1c5ad0431fee71766f0e5b99d523fd98a3b
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
  | PyTorch 1.11   | - |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 请用户自行获取SQuADv1.1数据集，可参照源码实现链接进行数据集获取，并将获取的数据集存放在源码包根目录下建立的**v1.1**目录下。

   ```
   mkdir v1.1
   cd v1.1
   ```

2. 数据集目录结构参考如下所示。

   ```
   ---源码包根目录
      ---v1.1
         ---train-v1.1.json
         ---dev-v1.1.json
         ---evaluate-v1.1.py
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

3. 下载词典

   在数据集v1.1目录下执行一下命令，建立**data/uncased_L-24_H-1024_A-16**文件目录。

   ```
   mkdir data
   cd data
   mkdir uncased_L-24_H-1024_A-16
   cd uncased_L-24_H-1024_A-16
   ```

   请用户自行下载**bert-base-uncased-vocab.txt**词典，并存放在**uncased_L-24_H-1024_A-16**文件夹目录下。

## 获取预训练模型
1. 获取预训练模型，用户可参考源码实现链接进行获取。
2. 确认预训练模型

    bert-large:
    ```
    ---bert_large_pretrained_amp.pt
    ```

    bert-base:

    ```
    ---bert_base.pt
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
     bash test/train_large_full_1p.sh --data_path=/xxx/v1.1 --ckpt_path=real_path  # bert-large单卡精度

     bash test/train_base_full_1p.sh --data_path=/xxx/v1.1 --ckpt_path=real_path   # bert-base单卡精度
     
     bash test/train_base_performance_1p.sh --data_path=/xxx/v1.1 --ckpt_path=real_path   # bert-base单卡性能
     ```
   
   - 单机8卡训练
   
     启动8卡训练。
   
     ```
     bash test/train_large_full_8p.sh --data_path=/xxx/v1.1 --ckpt_path=real_path  # bert-large 8卡精度

     bash test/train_base_full_8p.sh --data_path=/xxx/v1.1 --ckpt_path=real_path   # bert-base 8卡精度
     
     bash test/train_base_performance_8p.sh --data_path=/xxx/v1.1 --ckpt_path=real_path   # bert-base 8卡性能
     ```
   
   --data_path参数填写数据集路径，需写到数据集的v1.1目录。

   --ckpt_path为预训练模型存放路径，只需要传入文件所在目录即可，无需包含预训练模型文件名。

   --hf32开启HF32模式，不与FP32模式同时开启

   --fp32开启FP32模式，不与HF32模式同时开启

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --amp                               //是否使用混合精度
   --seed                              //随机数种子设定
   --learning_rate                     //学习率设置	  
   --train_batch_size                  //训练批次大小
   --npu_id                            //npu单卡id
   --addr                              //分布式训练地址
   --num_npu                           //npu训练卡使用个数
   --use_npu                           //是否使用npu
   --max_steps                         //训练的总步数
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME     | F1 |  FPS | Epochs | AMP_Type | Torch_Version |
| :-----:  | :--: | :--: | :--: | :--: | :--: |
| 1p-bert-large | - | 121 | 1 | O2 | 1.8 |
| 1p-bert-base | - | 333 | 1 | O2 | 1.8 |
| 8p-bert-large | 90.87 | 833 | 2 | O2 | 1.8 |
| 8p-bert-base | 87.011 | 2602 | 2 | O2 | 1.8 |

# 版本说明

## 变更

2022.02.17：更新readme，重新发布。

2021.08.01：首次发布。

## FAQ

第一次训练的第一个step特别慢，会对SQuAD做预处理，该过程非常耗时，通常需要十分钟左右。预处理完之后会在数据集相同目录下生成缓存文件，下次训练时会快很多。
