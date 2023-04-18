# Wav2Vec2.0 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述

## 简述

Wav2vec2.0是Meta在2020年发表的无监督语音预训练模型。它的核心思想是通过向量量化（Vector Quantization，VQ）构造自建监督训练目标，对输入做大量掩码后利用对比学习损失函数进行训练。

- 参考实现：
  
  ```
  url=https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec
  commit_id=a0ceabc287e26f64517fadb13a54c83b71e8e469
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
  | PyTorch 1.11   | - |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  apt-get install libsndfile1 (yum install libsndfile1)
  pip uninstall fairseq
  pip install -e ./
  ```
  若安装完后找不到fairseq-hydra-train，一般安装在`which python`命令所在的路径

## 准备数据集

1. 获取数据集。

   主要参考 [wav2vec2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) 进行 `LibriSpeech` 数据集准备。
   用户需自己新建一个 `$data_path` 路径，用于放预训练模型和数据集，`$data_path` 可以设置为服务器的任意目录（注意存放的磁盘需要为NVME固态硬盘）。
   下载 `LibirSpeed` 数据集，包括 `train-clean-100`，`dev-clean`，按照 [wav2vec2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) 准备 `manifest`，统一放置到 `$data_path` 目录下。
   数据集目录结构参考如下所示。
    ```
    $data_path
    ├── train-clean-100
    ├── dev-clean
    └── manifest
    ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

## 获取预训练模型

用户自行获取预训练模型，将获取的 `wav2vec_small.pt` 预训练模型放至在 `$data_path` 目录下。
 `$data_path` 最终的目录结构如下所示。
 ```
 $data_path
    ├── train-clean-100
    ├── dev-clean
    ├── wav2vec_small.pt
    └── manifest
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

     启动单卡训练

     ```
     bash ./test/train_full_1p.sh --data_path=$data_path  # 单卡精度
     bash ./test/train_performance_1p.sh --data_path=$data_path # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练

     ```
     bash ./test/train_full_8p.sh --data_path=$data_path  # 8卡精度
     bash ./test/train_performance_8p.sh --data_path=$data_path # 8卡性能
     ```
   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

      ```
      公共参数：
      --task.data                                     //数据集路径
      --hydra.run.dir                                 //hydra运行路径
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
| 1P-竞品V |   -   | 5524.7  | - | - | 1.5 |
| 8P-竞品V | 5.443  | 44493.3 | - | - | 1.5 |
| 1P-NPU |   -   | 4869.8  | - | - | 1.8 |
| 8P-NPU | 5.546 | 33463.9 | - | - | 1.8 |

# 版本说明

## 变更

2022.11.24：首次发布
2023.02.17：二次修正

## FAQ


无。