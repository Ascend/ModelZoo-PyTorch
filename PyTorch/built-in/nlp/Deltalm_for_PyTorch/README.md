# Deltalm for PyTorch

- [概述](概述.md)

- [准备训练环境](准备训练环境.md)

- [开始训练](开始训练.md)

- [训练结果展示](训练结果展示.md)

- [版本说明](版本说明.md)


# 概述

## 简述

Deltalm 模型是Fairseq套件中基于Transformer结构的翻译模型，在iwslt14 de2en数据集上训练和评估。

- 参考实现：

  ```
  url=https://github.com/microsoft/unilm/blob/master/deltalm
  commit_id=eb1cc35e63988b2fe8c1fae348012a57da096e43
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
  | PyTorch 1.8 | - |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装套件。

  在模型源码包根目录下执行以下命令。
  ```bash
  pip3.7 install -e ./fairseq
  ```
  安装相应库
  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

    1. 用户可参考源码GPU仓自行下载 `iwslt14` 数据集，并在预处理数据后，上传至到服务器任意目录中，如`/data-bin`
    2. 或者使用一键式处理工具`auto-data.sh`，需提前准备：
       1. tokenize模型："https://deltalm.blob.core.windows.net/deltalm/spm.model"
       2. 准备数据词典："https://deltalm.blob.core.windows.net/deltalm/dict.txt"
       3. 准备分词工具：参考"https://github.com/google/sentencepiece" readme操作安装`spm_encode `
       4. 执行脚本`bash auto-data.sh $1 $2 $3 $4 $5 $6`

          $1：原始数据生成目录 `/tmp/iwslt14`

          $2：最终处理数据目录 `/data-bin`

          $3：tokenize模型路径

          $4：词典路径
          
          $5: 数据预处理工具下载链接: [mosesdecoder](https://github.com/moses-smt/mosesdecoder.git)
          
          $6: 原始数据下载链接: [iwslt14](http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz)

2. 获取预训练模型
  用户自行下载`deltalm-base`预训练模型权重，并放置于上面预处理数据目录下
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

      ```bash
      bash ./test/train_performance_1p.sh --data_path=/data-bin  # 单卡性能
      ```

    - 单机8卡训练。

      启动8卡训练。

      ```bash
      bash ./test/train_full_8p.sh --data_path=/data-bin  # 8卡精度
      bash ./test/train_performance_8p.sh --data_path=/data-bin  # 8卡性能
      ```

      --data_path参数填写数据集路径，需写到数据集的一级目录。


    模型训练脚本参数说明如下。

    ```
    公共参数：
    --data_path                         //数据集路径
    --arch                              //使用模型架构
    --save-dir                          //权重文件保存路径
    --max-epoch                         //重复迭代轮数
    --max-tokens                        //最大token大小
    --lr                                //学习率
    --optimizer                         //使用哪种优化器
    --eval-bleu                         //使用评估指标
    --distributed-world-size            //是否进行分布式训练
    ```

    训练完成后，权重文件默认保存在当前路径的checkpoints目录下，test/out目录下并输出模型训练精度和性能信息。

# 训练结果展示

**表 3**  en_de数据集训练结果展示表

| NAME  | MODE | Bleu  | WPS  | Epochs | AMP_Type | Torch_Version |
| :---: |------|:-----:|:----:| :---: | :---: | :---: |
| 8p-竞品A | fp16 | 39.45 | 14401 | 100 | - | 1.8 |
| 8p-NPU | fp16 | 39.37 | 16214 | 100 | - | 1.8 |

> **说明：**
   >由于该模型默认开启二进制，所以在性能测试时，需要安装二进制包，安装方式参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。


# 版本说明

## 变更

2023.6.29：首次发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md