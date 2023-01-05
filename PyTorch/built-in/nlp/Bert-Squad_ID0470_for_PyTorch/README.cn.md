# Bert-Squad_ID0470_for_PyTorch

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
  | 硬件    | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 下载SQuADv1.1数据集：

```
mkdir v1.1
cd v1.1
下载数据集，参照：https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT
将数据集放到v1.1目录下
```

2. 确认数据集

```
   ---squad
      ---v1.1
         ---train-v1.1.json
         ---dev-v1.1.json
         ---evaluate-v1.1.py
```

3. 下载词典

在数据集v1.1目录执行

```
mkdir data/uncased_L-24_H-1024_A-16
cd data/uncased_L-24_H-1024_A-16
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt -O vocab.txt
```

## 获取预训练模型
1. 获取预训练模型，参照：https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT
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

参数说明：
data_path为v1.1数据集的路径,eg:/data/squad/v1.1
ckpt_path为预训练模型存放路径,只需要传入文件所在目录即可,无需包含与训练模型文件名.

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash test/train_large_full_1p.sh  --data_path=/data/squad/v1.1  --ckpt_path=real_path  #bert-large
     bash test/train_base_full_1p.sh  --data_path=/data/squad/v1.1  --ckpt_path=real_path   #bert-base
     ```
   
   - 单机8卡训练
   
     启动8卡训练。
   
     ```
     bash test/train_large_full_8p.sh  --data_path=/data/squad/v1.1  --ckpt_path=real_path  #bert-large
     bash test/train_base_full_8p.sh  --data_path=/data/squad/v1.1  --ckpt_path=real_path   #bert-large
     ```
   
   --data\_path参数填写数据集路径。

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

| NAME     | F1 |  FPS |
| -------  | :---:  | :--: |
| 1p-bert-large | - | - |
| 1p-bert-base | - | - |
| 8p-bert-large | 91.0 | 846 |
| 8p-bert-base | 86.97 | 2742 |

# 版本说明

## 变更

2022.09.01：更新pytorch1.8版本，重新发布。

2021.08.01：首次发布。

## 已知问题

第一次训练的第一个step特别慢，会对SQuAD做预处理，该过程非常耗时，通常需要十分钟左右。预处理完之后会在数据集相同目录下生成缓存文件，下次训练时会快很多。
