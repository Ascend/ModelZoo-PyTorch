# STGCN(ST-GCN) 模型推理指导

- [概述](#概述)
    - [输入输出数据](#输入输出数据)
- [推理环境](#推理环境)
- [快速上手](#快速上手)
    - [获取源码](#获取源码)
    - [准备数据集](#准备数据集)
    - [推理验证](#推理验证)
- [性能&精度](#性能精度)

----
# 概述

ST-GCN是一种图卷积神经网络，该模型可以实现对人体骨架图做姿态估计，从而实现行为识别的效果。该模型利用了人体骨骼的局部模式和相关特征信息。本项目利用昇腾推理引擎`AscendIE`和框架推理插件`TorchAIE`，基于`pytorch框架`实现该模型在昇腾设备上的高性能推理。

+ 论文  
    Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition

+ 参考实现  
    ```
    url = https://github.com/open-mmlab/mmskeleton/blob/master/mmskeleton/models/backbones/st_gcn_aaai18.py
    branch = master
    commit_id = b4c076baa9e02e69b5876c49fa7c509866d902c7
    model_name = ST_GCN_18
    ```
## 输入输出数据
+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | input1  |  FLOAT32  | ND         | batch_size x 3 x 300 x 18 x 2   |

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output1       |  FLOAT32   | ND          | batch_size x 400       |


----
# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套  | 版本  | 环境准备指导  |
  |---------| ------- | ------------------------------------------------------------ |
  | 固件与驱动 | 23.0.rc1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN | 7.0.RC1.alpha003 | - |
  | Python | 3.9.11 | - |
  | PyTorch | 2.0.1 | - |
  | Torch_AIE | 6.3.rc2 | - |

- 安装依赖

   ```
   pip install -r requirements.txt
   ```

----
# 快速上手

## 获取源码

1. 模型导出依赖st-gcn和mmskeleton仓库。
    ```bash
    git clone https://github.com/yysijie/st-gcn.git
    git clone https://github.com/open-mmlab/mmskeleton.git
    git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
    cp ModelZoo-PyTorch/ACL_PyTorch/built-in/cv/STGCN_for_Pytorch/stgcn_postprocess.py .
    ```


## 准备数据集

1. 获取原始数据集  
    参考[STGCN-Pytorch在昇腾设备上离线推理指导](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/STGCN_for_Pytorch)预处理数据，得到如下目录结构：
    ```
    data/
    |—— Kinetics/
        |—— kinetics-skeleton
            |—— val_label.pkl
            |—— ...
    |—— kinetics-skeleton
            |—— val_data
                |—— 0.bin
                |—— 1.bin
                ...
            |—— val_label
                |—— 0.bin
                |—— 1.bin
                ...
    ```

## 推理验证

1. 执行推理。

    ```
    python inference.py
    ```
    执行推理，并将计算结果保存在./result文件夹中。同时，推理结束后打印出性能数据。


2. 精度验证。

    调用脚本stgcn_postprocess.py，可以获得Accuracy数据。

    ```
    python stgcn_postprocess.py \
        --result_dir ./result \
        --label_path ./data/Kinetics/kinetics-skeleton/val_label.pkl
    ```

    - 参数说明：

    - result_dir：推理结果保存地址


----
# 性能&精度

在310P设备上，OM模型的精度与目标精的相对误差低于 1%，batchsize为1时模型性能最优，达 381.10 fps。

| 芯片型号   | BatchSize | 数据集      | 精度            | 性能       |
| --------- |---------- | ----------- | --------------- | --------- |
|Ascend310P3| 1         | kinetics-skeleton | Top1@Acc=31.61%   Top5@Acc: 53.74% | 329 fps |
|Ascend310P3| 4         | kinetics-skeleton | Top1@Acc=31.61%   Top5@Acc: 53.74% | 242 fps |
|Ascend310P3| 8         | kinetics-skeleton | Top1@Acc=31.61%   Top5@Acc: 53.74% | 243 fps |
|Ascend310P3| 16        | kinetics-skeleton | Top1@Acc=31.61%   Top5@Acc: 53.74% | 246 fps |
|Ascend310P3| 32        | kinetics-skeleton | Top1@Acc=31.61%   Top5@Acc: 53.74% | 247 fps |
|Ascend310P3| 64        | kinetics-skeleton | Top1@Acc=31.61%   Top5@Acc: 53.74% | 235 fps |

