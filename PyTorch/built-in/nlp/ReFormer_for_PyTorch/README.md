# ReFormer for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

Reformer: The Efficient Transformer是发表在ICLR 2020的一项工作，致力于用最小的内存训练Transformer网络。本文有两位共同一作，Nikita Kitaev和Łukasz Kaiser，后者是Attention is all you need作者之一。Reformer主要涉及了四处创新点：1. 使用Axial Positional Embedding来减小位置编码矩。2. 提出了基于局部敏感性哈希算法(Locality Sensitive Hashing, LSH)的self-attention机制。3. 提出可以使用分段的方式来处理全连接层(FFN)的计算过程，每次只计算一部分，从而不需要将整个序列加载到内存。4. 使用可逆(Reversible)残差连接来代替传统残差连接，好处是在前向过程中前N-1层的中间值和输出都不需要存储了，只保留最后一层的输出。本文档描述的Reformer是基于Pytorch实现的版本。


- 参考实现：

  ```
  url=https://github.com/huggingface/transformers.git
  commit_id=7378726df60b9cf399aacfe372fea629c1c4c7d3
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/nlp/
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.11 | transformers==4.28.1 |

  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r requirements.txt
  ```

- 替换transformers库中相关文件
  
  将源码包根目录下transformers_modify文件夹中的各个文件分别替换到transformers库安装目录下的对应位置（基于**transformers 4.28.1**版本）：
  
  ```shell
  training_args.py -> transformers/training_args.py
  trainer.py -> transformers/trainer.py
  trainer_pt_utils.py -> transformers/trainer_pt_utils.py
  modeling_reformer.py -> transformers/models/reformer/modeling_reformer.py
  ```

## 获取预训练模型
在源码包目录下新建文件夹checkpoint/reformer-crime-and-punishment，用户自行下载预训练模型至目录/checkpoint/reformer-crime-and-punishment下。Reformer预训练模型目录结构参考如下所示。
   ```
   ├── checkpoint
         ├──reformer-crime-and-punishment
              ├──config.json
              ├──generation_config.json
              ├──config.json
              ├──generation_config_for_text_generation.json
              ├──pytorch_model.bin
              ├──rust_model.ot
              ├──spiece.model
              ├──tokenizer.json
   ```

## 准备数据集

1. 获取数据集。

   在源码包目录下新建文件夹corpus，用户自行获取数据集放至目录corpus下。

   Reformer数据集目录结构参考如下所示。

   ```
   ├── corpus
         ├──train_corpus.txt                     
         ├──test_corpus.txt             
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。
  
2. 获取词表。
  在源码包目录下新建文件夹lf_token，用户自行获取词表，并放在目录lf_token下，目录结构如下。
   ```
   ├── lf_token
         ├──config.json
         ├──merges.txt
         ├──vocab.json
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
     bash ./test/train_full_1p.sh  # 单卡精度
     
     bash ./test/train_performance_1p.sh  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh  # 8卡精度
     
     bash ./test/train_performance_8p.sh  # 8卡性能
     ```

   训练完成后，权重文件保存在训练脚本指定的路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表
| NAME  | Acc@1  | FPS  | Epochs  | Torch_Version |
|:-:|:-:|:-:|:-:|:-:|
| 1p-竞品A  | - | 43.7914  | 1 | 1.11  |
| 8p-竞品A   | 0.1931  | 330.788  | 10  | 1.11  |
| 1p-NPU  | - | 30.6212  | 1 | 1.11  |
| 8p-NPU   | 0.1931  | 233.93  | 10  | 1.11  |

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md

# 版本说明

## 变更

2023.11.15：首次发布。

## FAQ

1. 在conda环境下运行报错：ImportError: libblas.so.3: cannot open shared object file: No such file or directory
```
conda install openblas blas-devel

conda install -c conda-forge blas
```

2. 报错ImportError: /home/miniconda3/envs/reformer/lib/python3.7/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block
```
find / -name "libgomp-d22c30c5.so.1.0.0"

export LD_PRELOAD=$LD_PRELOAD:/home/miniconda3/envs/reformer/lib/python3.7/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
```
