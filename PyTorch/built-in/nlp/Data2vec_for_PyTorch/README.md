# Data2vec for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述

## 简述

data2vec 是首个适用于多模态的高性能自监督算法。Meta AI 将 data2vec 分别应用于语音、图像和文本，在计算机视觉、语音任务上优于最佳单一用途算法，并且在 NLP 任务也能取得具有竞争力的结果。此外，data2vec 还代表了一种新的、全面的自监督学习范式，其提高了多种模态的进步，而不仅仅是一种模态。data2vec 不依赖对比学习或重建输入示例，除了帮助加速 AI 的进步，data2vec 让我们更接近于制造能够无缝地了解周围世界不同方面的机器。

- 参考实现：
  
  ```
  url=https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec
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
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |
  | PyTorch 1.11   | - |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip uninstall fairseq
  pip install -e ./
  pip install -r requirements.txt
  ```

## 准备数据集

1. 获取数据集。

   用户自行下载wikitext-103-raw-v1.zip数据集。参考examples/roberta/README.pretraining.md中的介绍进行数据集预处理。
   数据集目录结构参考如下所示。
    ```
    $data_path
    ├── dict.txt
    ├── preprocess.log
    ├── test.bin
    ├── test.idx
    ├── train.bin
    ├── train.idx
    ├── valid.bin
    └── valid.idx
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

   该模型支持单机单卡训练。

   - 单机单卡训练

     启动单卡训练

     ```
     bash ./test/train_full_1p.sh --data_path=$data_path  # 单卡精度
     bash ./test/train_performance_1p.sh --data_path=$data_path  # 单卡性能
     ```

   - 单机单卡评测

     启动单卡评测

     ```
     bash ./test/train_eval_1p.sh --data_path=$data_path --checkpoint_path=$checkpoint_path  # 单卡评测
     ```

   模型训练脚本参数说明如下。

      ```
      公共参数：
      --task.data                                     //数据集路径
      --distributed_training.distributed_world_size   //训练设备数量
      --optimization.max_update                       //优化器最大更新次数
      --config-dir                                    //配置文件路径
      --config-name                                   //配置文件名称
      ```
    
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

|  Name  | wer  | FPS | Epochs | AMP_Type | Torch_Version |
| :----: | :---: | :--: |:----: | :---: | :--: |
| 1P-竞品V |   -   | -  | - | - | 1.8 |
| 1P-NPU |   -   | -  | - | - | 1.8 |

# 版本说明

## 变更

2023.05.30：首次发布。

## FAQ


无。
