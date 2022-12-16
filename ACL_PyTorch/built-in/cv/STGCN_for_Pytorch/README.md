# STGCN(ST-GCN) 模型推理指导

- [概述](#概述)
    - [输入输出数据](#输入输出数据)
- [推理环境](#推理环境)
- [快速上手](#快速上手)
    - [获取源码](#获取源码)
    - [准备数据集](#准备数据集)
    - [模型转换](#模型转换)
    - [推理验证](#推理验证)
- [性能&精度](#性能精度)

----
# 概述

ST-GCN是一种图卷积神经网络，该模型可以实现对人体骨架图做姿态估计，从而实现行为识别的效果。该模型利用了人体骨骼的局部模式和相关特征信息。
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
# 推理环境

- 该模型推理所需配套的软件如下：

    | 配套      | 版本    | 环境准备指导 |
    | --------- | ------- | ---------- |
    | 固件与驱动 | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN      | 6.0.RC1 | -          |
    | Nvidia-Driver | 460.67 |         |
    | CUDA      | 10.0    | -          |
    | CUDNN     | 7.6.5.32 | -          |
    | Python    | 3.7.5   | -          |
    
    说明：请根据推理卡型号与 CANN 版本选择相匹配的固件与驱动版本。


----
# 快速上手

## 获取源码

1. 在GPU服务器上安装`CUDA`与`CUDNN`（版本参照上表），然后依次执行以下命令安装python第三方库。
    ```bash
    conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
    pip install mmcv==0.4.3
    pip install Cython==0.29.32
    git clone https://github.com/open-mmlab/mmdetection.git
    cd ./mmdetection
    git checkout master
    git reset --hard 4357697acaaf7b3eb17a9e78f2e0b8996bcf4e73
    python setup.py install
    cd ..
    git clone https://github.com/open-mmlab/mmskeleton.git
    cd mmskeleton
    git checkout master
    git reset --hard b4c076baa9e02e69b5876c49fa7c509866d902c7
    python setup.py develop
    cd mmskeleton/ops/nms
    python setup_linux.py develop
    cd -
    pip install Pillow==6.2.2
    ```
    执行完后，把模型推理的业务代码与补丁文件都复制到当前目录。

2. 修改源码  
    PyTorch版本太过老旧，`torch.einsum` 与 `F.avg_pool2d` 不能映射到ONNX的算子集，所以执行以下命令修改源码：
    ```bash
    patch -p1 < stgcn.patch
    ```


## 准备数据集

1. 获取原始数据集  
    该模型使用`Kinetics-skeleton`行为识别数据集来验证模型精度。从[开源链接](https://drive.google.com/open?id=103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb)下载数据集。该推理业务需要的数据以及目录结构如下：
    ```
    data/
    `-- Kinetics/
        `-- kinetics-skeleton
            |-- val_data.npy
            `-- val_label.pkl
    ```


2. 数据预处理  
    执行前处理脚本将原始数据转换为OM模型输入需要的bin/npy文件。
    ```bash
    python stgcn_preprocess.py \
        --data_path ./data/Kinetics/kinetics-skeleton/val_data.npy \
        --label_path ./data/Kinetics/kinetics-skeleton/val_label.pkl \
        --output_dir ./data/kinetics-skeleton/
    ```
    参数说明：
    + --data_path: 原始数据存放位置
    + --label_path: 原始标签存放位置
    + --output_dir: 输出文件的保存目录
    
    运行成功后，data/kinetics-skeleton/目录下会创建val_data和val_label两个子目录，每个子目录下面都生成19796个bin文件。


## 模型转换

1. PyTroch 模型转 ONNX 模型  
    ```bash
    python stgcn_pth2onnx.py --ckpt ./checkpoints/st_gcn.kinetics-6fa43f73.pth --onnx ./st_gcn.onnx
    ```
    参数说明：
    + --ckpt: 预训练权重文件的路径
    + --onnx: 生成ONNX模型的保存路径

2. ONNX 模型转 OM 模型  
    此步骤只能在NPU设备上进行，所以执行atc命令转换模型前，需将上一步生成的ONNX复制到NPU设备。

    step1: 查看NPU芯片名称 \${chip_name}
    ```bash
    npu-smi info
    ```
    例如该设备芯片名为 310P3，回显如下：
    ```
    +-------------------+-----------------+------------------------------------------------------+
    | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
    | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
    +===================+=================+======================================================+
    | 0       310P3     | OK              | 15.8         42                0    / 0              |
    | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
    +===================+=================+======================================================+
    | 1       310P3     | OK              | 15.4         43                0    / 0              |
    | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
    +===================+=================+======================================================+
    ```

    step2: ONNX 模型转 OM 模型
    ```bash
    # 配置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /etc/profile
    
    bs=1
    chip_name=310P3
    
    # 执行 ATC 进行模型转换
    atc --model=st_gcn.onnx \
        --framework=5 \
        --output=st_gcn_bs${bs} \
        --input_format=ND \
        --input_shape="input1:${bs},3,300,18,2" \
        --soc_version=Ascend310${chip_name}
    ```

   参数说明：
    + --framework: 5代表ONNX模型
    + --model: ONNX模型路径
    + --input_shape: 模型输入数据的shape
    + --input_format: 输入数据的排布格式
    + --output: OM模型路径，无需加后缀
    + --soc_version: 处理器型号


## 推理验证

1. 该离线模型使用ais_infer作为推理工具，请参考[**安装文档**](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer#%E4%B8%80%E9%94%AE%E5%AE%89%E8%A3%85)安装推理后端包aclruntime与推理前端包ais_bench。完成安装后，执行以下命令预处理后的数据进行推理。
    ```bash
    python -m ais_bench
        --model ./st_gcn_bs${bs}.om \
        --input ./data/kinetics-skeleton/ \
        --output ./ \
        --output_dirname ./st_gcn_bs${bs}_out
        --batchsize ${bs}
    ```
    参数说明：
    + --model: OM模型路径
    + --input: 存放预处理后数据的目录路径
    + --output: 用于存放推理结果的父目录路径
    + --output_dirname: 用于存放推理结果的子目录路径，位于--output指定的目录下
    + --batchsize: 模型一次处理多少样本

2. 性能验证  

    对于性能的测试，需要注意以下三点：
    + 测试前，请通过`npu-smi info`命令查看NPU设备状态，请务必在NPU设备空闲的状态下进行性能测试。
    + 为了避免测试过程因持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps，反映模型在单位时间（1秒）内处理的样本数。
    ```bash
    python -m ais_bench --model ./st_gcn_bs${bs}.om --loop 100 --batchsize ${bs}
    ```
    
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  
    
    此步骤需要将NPU服务器上OM模型的推理结果复制到GPU服务器上，然后再GPU服务器上执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python stgcn_postprocess.py \
        --result_dir ./st_gcn_bs${bs}_out/ \
        --label_path ./data/Kinetics/kinetics-skeleton/val_label.pkl
    ```
    参数说明：
    + --result_dir: 存放推理结果的目录路径
    + --label_path: 标签文件所在路径
    
    运行成功后，程序会打印出模型的精度指标：
    ```
    Top 1: 31.59%
    Top 5: 53.74%
    ```


----
# 性能&精度

在310P设备上，OM模型的精度与目标精的相对误差低于 1%，batchsize为1时模型性能最优，达 381.10 fps。

| 芯片型号   | BatchSize | 数据集      | 精度            | 性能       |
| --------- |---------- | ----------- | --------------- | --------- |
|Ascend310P3| 1         | kinetics-skeleton | Top1@Acc=31.59%   Top5@Acc: 53.74% | 381.10 fps |
|Ascend310P3| 4         | kinetics-skeleton | Top1@Acc=31.59%   Top5@Acc: 53.74% | 237.11 fps |
|Ascend310P3| 8         | kinetics-skeleton | Top1@Acc=31.59%   Top5@Acc: 53.74% | 222.46 fps |
|Ascend310P3| 16        | kinetics-skeleton | Top1@Acc=31.59%   Top5@Acc: 53.74% | 217.65 fps |
|Ascend310P3| 32        | kinetics-skeleton | Top1@Acc=31.59%   Top5@Acc: 53.74% | 223.21 fps |
|Ascend310P3| 64        | kinetics-skeleton | Top1@Acc=31.59%   Top5@Acc: 53.74% | 219.39 fps |

