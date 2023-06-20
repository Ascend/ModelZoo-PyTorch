# Fairseq Transformer wmt18 for PyTorch

- [概述](概述.md)

- [准备训练环境](准备训练环境.md)

- [开始训练](开始训练.md)

- [训练结果展示](训练结果展示.md)

- [版本说明](版本说明.md)


# 概述

## 简述

Fairseq Transformer wmt18模型是Fairseq套件中基于Transformer结构的翻译模型，在wmt18 en2de数据集上训练和评估。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/fairseq
  commit_id=3f6ba43f07a6e9e2acf957fc24e57251a7a3f55c
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
  | PyTorch 1.11 | - |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装套件。

  在模型源码包根目录下执行以下命令。
  ```bash
  pip3.7 install -e ./ 
  ```


## 准备数据集

1. 获取数据集。
  
    用户自行下载 `WMT18` (Workshop on Machine Translation at EMNLP 2018)数据集，并将[表2 数据集简介表](表2)中的数据集全部下载后上传到源码包中的 `./examples/translation/orig` 目录下并解压。

    **表 2**  数据集简介表
    | 来源  | 名称                                             |
    | :---- |:----------------------------------------------- |
    | wmt13 | [training-parallel-europarl-v7.tgz](http://statmt.org/wmt13/training-parallel-europarl-v7.tgz) |
    | wmt13 | [training-parallel-commoncrawl.tgz](http://statmt.org/wmt13/training-parallel-commoncrawl.tgz) |
    | wmt18 | [paracrawl-release1.en-de.zipporah0-dedup-clean.tgz](https://s3.amazonaws.com/web-language-models/paracrawl/release1/paracrawl-release1.en-de.zipporah0-dedup-clean.tgz) |
    | wmt18 | [training-parallel-nc-v13.tgz](http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz) |
    | wmt18 | [dev.tgz](http://data.statmt.org/wmt18/translation-task/dev.tgz) |
    | wmt18 | [test.tgz](http://data.statmt.org/wmt18/translation-task/test.tgz) |

    其中，前五项语料为训练集+验证集；最后一项语料为测试集。

2. 下载tokenize和分词工具。

    进入数据集目录执行以下命令。
    ```bash
    cd examples/translation/
    echo 'Cloning Moses github repository (for tokenization scripts)...'
    git clone https://github.com/moses-smt/mosesdecoder.git
    echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
    git clone https://github.com/rsennrich/subword-nmt.git
    cd ../../
    ```

2. 数据预处理。

    进入数据集目录运行预处理脚本。该阶段需要进行分词，耗时较长，请耐心等待。
    ```bash
    cd examples/translation/
    bash prepare-wmt18en2de.sh
    cd ../../
    ```

3. 生成训练数据集。
  
    进入源码包根目录下执行以下命令。
    ```bash
    TEXT=examples/translation/wmt18_en_de
    fairseq-preprocess \
      --source-lang en --target-lang de \
      --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
      --thresholdtgt 0 --thresholdsrc 0 \
      --workers 20 \
      --destdir data-bin/wmt18_en_de  # 数据集路径，可根据实际情况进行调整
    ```

# 开始训练

## 训练模型

1. 进入源码包根目录。

   ```bash
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

    该模型支持单机单卡训练和单机8卡训练。

    - 单机单卡训练

      启动单卡训练。

      fp16训练命令
      ```bash
      bash ./test/train_performance_1p_fp16.sh --data_path=data-bin/wmt18_en_de  # 单卡性能
      ```
      
      fp32训练命令
      ```bash
      bash ./test/train_performance_1p_fp32.sh --data_path=data-bin/wmt18_en_de  # 单卡性能
      ```

    - 单机8卡训练。

      启动8卡训练。

      fp16训练命令
      ```bash
      bash ./test/train_full_8p_fp16.sh --data_path=data-bin/wmt18_en_de  # 8卡精度
      bash ./test/train_performance_8p_fp16.sh --data_path=data-bin/wmt18_en_de  # 8卡性能
      ```
      
      fp32训练命令
      ```bash
      bash ./test/train_full_8p_fp32.sh --data_path=data-bin/wmt18_en_de  # 8卡精度
      bash ./test/train_performance_8p_fp32.sh --data_path=data-bin/wmt18_en_de  # 8卡性能
      ```

      data_path为数据集路径，路径写到wmt18_en_de。

    
    模型训练脚本参数说明如下。

    ```
    公共参数：
    --data_path                         //数据集路径
    --save-dir                          //权重文件保存路径
    --max-epoch                         //重复迭代轮数
    --max-tokens                        //最大token大小
    --lr                                //学习率
    --distributed-world-size            //是否进行分布式训练
    ```
    
    训练完成后，权重文件默认保存在当前路径的checkpoints/transformer_wmt_en_de目录下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 3**  en_de数据集训练结果展示表

| NAME  | MODE | Bleu  | WPS  | Epochs | AMP_Type | Torch_Version |
| :---: |------|:-----:|:----:| :---: | :---: | :---: |
| 8p-竞品A | fp16 | 41.14 | 450k | 20 | - | 1.11 |
| 8p-NPU | fp16 | 41.17 | 170k | 20 | - | 1.11 |
| 8p-竞品A | fp32 | 41.12 | 334k | 20 | - | 1.11 |
| 8p-NPU | fp32 | 41.21 | 223k | 20 | - | 1.11 |

> **说明：** 
   >由于该模型默认开启二进制，所以在性能测试时，需要安装二进制包，安装方式参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md

# 版本说明

## 变更

2023.6.9：首次发布。

## FAQ

无。
