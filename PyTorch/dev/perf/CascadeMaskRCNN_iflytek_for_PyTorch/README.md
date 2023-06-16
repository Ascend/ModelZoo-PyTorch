# CascadeMaskRCNN for PyTorch

- Overview

- Prepare Training Environment

- Start Training

- Training Results Display

- Version Description

# Overview

## Introduction

The CascadeMaskRCNN model is an instance segmentation model provided by MMDetection, trained and evaluated on the COCO2017 dataset.

- Reference implementation：

  ```
  url=https://github.com/open-mmlab/mmdetection/tree/v2.13.0
  ```

- Implementation adapted to Ascend：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/dev/perf/
  ```


# Prepare Training Environment

## Prepare Environment

- The PyTorch version and known third-party library dependencies supported by the current model are shown in the table below.

  **Table 1**  Version Support Table

  | Torch_Version      | Third-party Dependency Version                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.11 | MMCV 1.x |

- Environment preparation guide

  For accurate PyTorch environment, please refer to https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes

  For MMCV environment preparation, please refer to https://github.com/open-mmlab/mmcv/blob/1.x/docs/zh_cn/get_started/build.md


## Prepare Dataset

  
  Users download the [COCO2017](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset/)dataset and unzip it themselves, and modify the dataset root path in configs/base/datasets/coco_instance.py according to the path after unzipping.


# Start Training

## Train Model

1. Run the training script.

    This model supports single-machine single-card training and single-machine 8-card training.

    - Single-machine single-card training

      Start single-card training.

      ```bash
      bash ./test/train_performance_1p.sh # Single card performance
      ```

    - Single-machine 8-card training.

      Start 8-card training.

      ```bash
      bash ./test/train_full_8p.sh  # 8-card accuracy
      bash ./test/train_performance_8p.sh  # 8-card performance
      ```
    
    After training is completed, the weight file and log information are saved by default in the CascadeMaskRCNN_iflytek_for_PyTorch directory under the current path.

# Training Results Display

**Table 2**  COCO2017 dataset training results display table
| NAME  | Batch_Size  | Epochs | Throughput | Box AP | Mask AP | AMP_Type | Torch_Version |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 8p-Competitor A | 2 | 12 | 40.51 | 41.2 | 36.0 | - | 1.11 |
| 8p-NPU | 2 | 12 | 20.28 | 41.3 | 35.8 | - | 1.11 |

> **Note：** 
Since this model defaults to binary, you need to install the binary package when testing performance, and the installation method refers to https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes

# Version Description

## Changes

2023.6.15：First release.

## FAQ

无。