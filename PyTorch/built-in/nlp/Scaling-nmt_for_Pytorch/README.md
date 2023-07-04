# Scaling-nmt for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述

## 简述

Scaling-NMT是一种用于神经机器翻译（NMT）的模型架构，旨在解决NMT中的可扩展性问题。在传统的NMT模型中，随着数据集的增大，模型的大小和计算量也会增加，导致训练和推理时间变得非常长。Scaling-NMT通过使用分层架构和动态路由机制来解决这个问题。在Scaling-NMT中，模型被分为多个层次，每个层次包含多个子模型。每个子模型只处理输入的一部分，然后将其传递给下一个子模型。这种分层架构可以使模型更容易扩展，因为每个子模型的大小和计算量都比整个模型小得多。

- 参考实现：
  
  ```
  url=https://github.com/facebookresearch/fairseq/tree/v0.12.2/examples/scaling_nmt
  commit_id=4a388e64cd646ed7d7ad8de8fae55df2b8eea91d
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
  | PyTorch 1.8 | torchvision==0.9.1 |
  | PyTorch 1.11   | - |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip uninstall fairseq
  pip install -e ./
  pip install -r 1.8_requirements.txt
  ```

## 准备数据集

1. 获取数据集。

   用户自行下载WMT'16 En-De数据集。在源码包根目录下新建 wmt16_en_de_bpe32k 文件夹，将数据集上传到 wmt16_en_de_bpe32k 目录下并解压。
   
2. 数据预处理。
   在源码包根目录下新建 data-bin 文件夹，并执行以下命令进行数据预处理。
   ```
   fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref wmt16_en_de_bpe32k/train.tok.clean.bpe.32000 \
    --validpref wmt16_en_de_bpe32k/newstest2013.tok.bpe.32000 \
    --testpref wmt16_en_de_bpe32k/newstest2014.tok.bpe.32000 \
    --destdir data-bin/wmt16_en_de_bpe32k \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary \
    --workers 20
   ```
   预处理后，数据集目录结构参考如下所示。
    ```
    |——data-bin
        |——wmt16_en_de_bpe32k
            ├── dict.de.txt
            ├── dict.en.txt
            ├── preprocess.log
            ├── test.en-de.de.bin
            ├── test.en-de.de.idx
            ├── test.en-de.en.bin
            ├── test.en-de.en.idx
            ├── train.en-de.de.bin
            ├── train.en-de.de.idx
            ├── train.en-de.en.bin
            ├── train.en-de.en.idx
            ├── valid.en-de.de.bin
            ├── valid.en-de.de.idx
            ├── valid.en-de.en.bin
            └── valid.en-de.en.idx
    ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练

     ```
     bash ./test/train_performance_1p.sh --data_path=$data_path  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练

     ```
     bash ./test/train_full_8p.sh --data_path=$data_path  # 8卡性能
     bash ./test/train_performance_8p.sh --data_path=$data_path  # 8卡性能
     ```

   模型训练脚本参数说明如下。

      ```
      公共参数：
      --arch                                     //模型架构
      --optimizer                                //优化器
      --adam-betas                               //优化器参数
      --lr                                       //初始学习率
      --warmup-updates                           //预热训练更新次数
      --dropout                                  //dropout参数
      --weight-decay                             //权重衰减
      --criterion                                //损失计算方法
      --max-tokens                               //最大tokens
      --fp16                                     //是否使用fp16
      --keep-last-epochs                         //保存最后几个epoch的权重
      --distributed-world-size                   //训练卡数量
      --device-id                                //指定训练卡
      --max-update                               //最大训练迭代次数
      ```
    
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

3. 模型评估。
   - 首先，使用 average_checkpoints.py 脚本对最后几个 checkpoint 求平均值。对最后 5-10 个 checkpoint 求平均通常很好，但您可能需要根据训练时间进行调整。
     ```
     python scripts/average_checkpoints \
        --inputs /path/to/checkpoints \
        --num-epoch-checkpoints 10 \
        --output checkpoint.avg10.pt
     ```
  
   - 接下来，使用 bean=4, lenpen=0.6 生成转换数据。
     ```
     fairseq-generate \
        data-bin/wmt16_en_de_bpe32k \
        --path checkpoint.avg10.pt \
        --beam 4 --lenpen 0.6 --remove-bpe > gen.out
     ```

   - 最后，计算 BLEU 指标。
     ```
     bash scripts/compound_split_bleu.sh gen.out
     ```

# 训练结果展示

**表 2**  训练结果展示表

|  Name  | BLEU  | WPS | max-update | MODE | Torch_Version |
| :----: | :---: | :--: |:----: | :--: | :--: |
| 1P-竞品V |   -   | 22743.4 | 1000 | fp16 | 1.8 |
| 8P-竞品V | 28.33 | 110441 | 300000 | fp16 | 1.8 |
| 1P-NPU |   -   | 17729.7 | 1000 | fp16 | 1.8 |
| 8P-NPU | 28.47 | 84627.5 | 300000 | fp16 | 1.8 |

# 版本说明

## 变更

2023.07.04：首次发布。

## FAQ

无。
