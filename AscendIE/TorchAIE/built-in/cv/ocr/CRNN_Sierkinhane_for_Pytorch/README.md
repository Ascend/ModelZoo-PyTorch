# CRNN_Sierkinhane 推理指导

- [概述](#概述)
  - [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
    - [获取源码](#获取源码)
    - [准备数据集](#准备数据集)
    - [编译模型](#编译模型)
    - [执行推理](#执行推理)
- [精度和性能](#精度和性能)

## 概述

CRNN_Sierkinhane 是一个基于卷积循环网络的中文 OCR 模型。
参考实现：

```
https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec
branch=stable
commit_id=a565687c4076b729d4059593b7570dd388055af4
```

## 输入输出数据

| 输入数据 | 数据类型 | 大小 | 数据排布格式 |
| ---- | ---- | ---- | ---- |
| input | FLOAT32  | <batch_size> x 1 x 32 x 160 | NCHW |

| 输出数据 | 数据类型 | 大小 | 数据排布格式 |
| ---- | ---- | ---- | ---- |
| output | FLOAT32 | 41 x <batch_size> x <字符集总数 + 1> | ND |

数据集：GitHub 仓库提供的 360 万数据集。

## 推理环境准备

- 该模型需要以下固件与插件

  **表 1**  版本配套表

| 配套                                                            | 版本    | 
| ------------------------------------------------------------    | ------- | 
| 固件与驱动                                                       | 23.0.RC1  | 
| CANN                                                            | 7.0.RC1.alpha003 | 
| Python                                                          | 3.9.11   | 
| PyTorch                                                         | 2.0.1 | 
| Torch_AIE                                                       | 6.3.rc2 |
| 芯片类型                                                         | Ascend310P3 |


## 快速上手
### 获取源码
1. 拉取Gitee仓代码，并安装依赖。
   ```bash
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/AscendIE/TorchAIE/built-in/cv/ocr/CRNN_Sierkinhane_for_Pytorch
   pip install -r requirements.txt
   ```

2. 克隆 GitHub 仓库，切换到指定分支、指定 commit_id。
   ```bash
   git clone https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec.git
   cd CRNN_Chinese_Characters_Rec
   git checkout stable
   git reset --hard a565687c4076b729d4059593b7570dd388055af4
   cd ..
   ```
3. 将本仓代码拷贝至CRNN_Chinese_Characters_Rec目录下。
   ```bash
   cp aie_compile.py ./CRNN_Chinese_Characters_Rec
   cp aie_val.py ./CRNN_Chinese_Characters_Rec
   cp config.py ./CRNN_Chinese_Characters_Rec
   cd ./CRNN_Chinese_Characters_Rec
   ```

### 准备数据集

1. 请按照 GitHub 仓中提供的方式下载原始数据集和label，并遵守数据集提供方要求使用。下载好原始数据集并解压后，更改/lib/config/360CC_config.yaml文件中的对应内容。

   ```bash
   DATASET:
      ROOT: 'to/your/images/path'
   ```

### 编译模型

1. 执行模型编译脚本，使用torch_aie编译模型使其可以运行在昇腾npu上。(以bs8为例)

   ```bash
   python aie_compile.py --batch_size=8
   ```
   参数说明：

   - --batch_size：批大小。

### 执行推理

1. 执行推理脚本，获得在数据集上的模型准确度与吞吐量。(以bs8为例)

   ```bash
   python aie_val.py --cfg=./lib/config/360CC_config.yaml --batch_size=8 --model_path=./crnn_sierkinhane_bs8.pt
   ```

   参数说明：
   - --cfg：360CC_config.yaml文件路径。
   - --batch_size：批大小。
   - --model_path：编译后模型的路径。


## 精度和性能

1. 纯静态输入

| 芯片型号 | Batch Size | 数据集 | ACC-精度 | 性能 |
| ---- | ---- | ---- | ---- | ----|
| 310P3 | 1 | GitHub 仓库提供的 360 万数据集 | 78.37% | 583.00 |
| 310P3 | 4 | GitHub 仓库提供的 360 万数据集 | - | 2172.97 |
| 310P3 | 8 | GitHub 仓库提供的 360 万数据集 | - | 3622.37 |
| 310P3 | 16 | GitHub 仓库提供的 360 万数据集 | - | 5425.79 |
| 310P3 | 32 | GitHub 仓库提供的 360 万数据集 | - | 6864.79 |
| 310P3 | 64 | GitHub 仓库提供的 360 万数据集 | - | 7608.74 |
