# CascadeMaskRCNN for PyTorch

- 概述

- 准备训练环境

- 开始训练

- 训练结果展示

- 版本说明


# 概述

## 简述

CascadeMaskRCNN模型是MMDetection中提供的实例分割模型，在COCO2017数据集上训练和评估。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection/tree/v2.13.0
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/dev/perf/
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.11 | MMCV 1.x |

- 环境准备指导。

  PyTorch环境准确请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》

  MMCV环境准备请参考《[从源码编译 MMCV](https://github.com/open-mmlab/mmcv/blob/1.x/docs/zh_cn/get_started/build.md)》


## 准备数据集

  
  用户自行下载 [COCO2017](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset/)数据集并解压，根据解压后的路径修改configs/_base_/datasets/coco_instance.py下的数据集根路径。


# 开始训练

## 训练模型

1. 运行训练脚本。

    该模型支持单机单卡训练和单机8卡训练。

    - 单机单卡训练

      启动单卡训练。

      ```bash
      bash ./test/train_performance_1p.sh # 单卡性能
      ```

    - 单机8卡训练。

      启动8卡训练。

      ```bash
      bash ./test/train_full_8p.sh  # 8卡精度
      bash ./test/train_performance_8p.sh  # 8卡性能
      ```
    
    训练完成后，权重文件与日志信息默认保存在当前路径的CascadeMaskRCNN_iflytek_for_PyTorch目录下。

# 训练结果展示

**表 2**  COCO2017数据集训练结果展示表
| NAME  | Batch_Size  | Epochs | Throughput | Box AP | Mask AP | AMP_Type | Torch_Version |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 8p-竞品A | 2 | 12 | 40.51 | 41.2 | 36.0 | - | 1.11 |
| 8p-NPU | 2 | 12 | 20.28 | 41.3 | 35.8 | - | 1.11 |

> **说明：** 
   >由于该模型默认开启二进制，所以在性能测试时，需要安装二进制包，安装方式参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

# 版本说明

## 变更

2023.6.15：首次发布。

## FAQ

无。