# RepVGG 模型-推理指导

- [概述](#概述)
  - [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)

- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型转换](#模型转换)
  - [推理验证](#推理验证)

- [精度&性能](#精度性能)


# 概述

RepVGG是一个分类网络，该网络是在VGG网络的基础上进行改进，主要的改进点包括：
   1. 在VGG网络的Block块中加入了Identity和残差分支，相当于把ResNet网络中的精华应用 到VGG网络中；
   2. 模型推理阶段，通过Op融合策略将所有的网络层都转换为 Conv3*3，便于模型的部署与加速

+ 论文  
    [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)  
    Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, Jian Sun

+ 参考实现  
    ```
    url = https://github.com/DingXiaoH/RepVGG
    branch = main
    commit_id = 9f272318abfc47a2b702cd0e916fca8d25d683e7
    ```

## 输入输出数据

+ 输入数据

    | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
    | -------- | -------- | ------------------------- | ------------ |
    | actual_input_1 | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


+ 输出数据

    | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
    | -------- | -------- | -------- | ------------ |
    | output1  | batchsize x 1000 | FLOAT32  | ND           |



# 推理环境准备

+ 该模型需要以下插件与驱动

  **表 1**  版本配套表

    | 配套                                                         | 版本    | 环境准备指导                                                 |
    | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
    | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN                                                         | 6.0.RC1 | -                                                            |
    | Python                                                       | 3.8.13  | -                                                            |
    | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |


# 快速上手

## 获取源码

1. 安装依赖。
    ```bash
    pip install -r requirements.txt
    ```

2. 获取源码。

    在当前目录下，执行如下命令。

    ```
    git clone https://github.com/DingXiaoH/RepVGG
    cd RepVGG
    git checkout main
    git reset --hard 9f272318abfc47a2b702cd0e916fca8d25d683e7
    cd ..        
    ```
    

## 准备数据集

1. 获取原始数据集  
    本模型推理项目使用 ILSVRC2012 数据集验证模型精度，请在 [ImageNet官网](https://gitee.com/link?target=http%3A%2F%2Fimage-net.org%2F) 自行下载ILSVRC2012数据集并解压，本模型将用到 ILSVRC2012_img_val.tar 验证集及 ILSVRC2012_devkit_t12.gz 中的 val_label.txt 标签文件。
    
    请按以下的目录结构存放数据：
    ```
    ├── imageNet/
        ├── val/
            ├──ILSVRC2012_val_00000001.JPEG
            ├──ILSVRC2012_val_00000002.JPEG
            ├──...
        ├── val_label.txt
    ```

2. 数据预处理  
    执行前处理脚本将原始数据转换为OM模型输入需要的bin文件。
    ```bash
    python RepVGG_preprocess.py --src_path /opt/npu/imageNet/val --save_path ./prep_dataset
    ```
    参数说明：
    + --src_path: 测试图片所在的目录路径
    + --save_path: 存放生成的bin文件的目录路径
    
    运行成功后，每张原始图片都会对应生成一个bin文件存放于 ./prep_dataset 目录下，总计50000个bin文件。


## 模型转换

1. PyTorch模型转ONNX模型  
    进入 [GoogleDrive](https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq?usp=sharing) 或 [BaiduCloud(rvgg)](https://pan.baidu.com/s/1nCsZlMynnJwbUBKn0ch7dQ)，链接里包含多个预训练模型，只需下载 RepVGG-A0-train.pth。然后执行执行以下命令生成 ONNX 模型：
     ```bash
     python RepVGG_pth2onnx.py --checkpoint RepVGG-A0-train.pth --onnx RepVGG.onnx
     ```
    参数说明：
    + --checkpoint: 预训练权重文件的路径
    + --onnx: 生成ONNX模型的保存路径

    运行成功后，即可获得“RepVGG.onnx”文件。

2. 使用ATC工具将ONNX模型转为OM模型

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
    
    chip_name=310P3  # 根据 step1 的结果设值
    bs=1  # 根据需要自行设置 

    atc --framework=5 \
        --model=RepVGG.onnx \
        --output=RepVGG_bs${bs} \
        --input_format=NCHW \
        --input_shape="actual_input_1:${bs},3,224,224" \
        --log=error \
        --soc_version=Ascend${chip_name}
    ```

   参数说明：
    + --framework: 5代表ONNX模型
    + --model: ONNX模型路径
    + --input_shape: 模型输入数据的shape
    + --input_format: 输入数据的排布格式
    + --output: OM模型路径，无需加后缀
    + --log：日志级别
    + --soc_version: 处理器型号
    + --insert_op_conf：插入算子的配置文件


## 推理验证

1. 对数据集推理  
    该离线模型使用ais_infer作为推理工具，请参考[**安装文档**](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer#%E4%B8%80%E9%94%AE%E5%AE%89%E8%A3%85)安装推理后端包aclruntime与推理前端包ais_bench。完成安装后，执行以下命令预处理后的数据进行推理。  
    ```bash
    python -m ais_bench \
        --model ./RepVGG_bs${bs}.om \
        --input ./prep_dataset \
        --output ./ \
        --output_dirname lcmout/ \
        --outfmt NPY \
        --batchsize ${bs}
    ```
    参数说明：
    + --model OM模型路径
    + --input 存放预处理后数据的目录路径
    + --output 用于存放推理结果的父目录路径
    + --output_dirname 用于存放推理结果的子目录名，位于--output指定的目录下
    + --outfmt 推理结果文件的保存格式
    + --batchsize 模型每次输入bin文件的数量

2. 性能验证  
    对于性能的测试，需要注意以下三点：
    + 测试前，请通过`npu-smi info`命令查看NPU设备状态，请务必在NPU设备空闲的状态下进行性能测试。
    + 为了避免测试过程因持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps，反映模型在单位时间（1秒）内处理的样本数。
    ```bash
    python -m ais_bench --model ./RepVGG_bs${bs}.om --batchsize ${bs}
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证

    执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python RepVGG_postprocess.py \
        --result_path ./lcmout/_summary.json \
        --gtfile_path /opt/npu/imageNet/val_label.txt
    ```
    参数说明：
    + --result_path：生成推理结果summary.json所在路径。
    + --gtfile_path：标签val_label.txt所在路径
    
# 精度&性能

在310P设备上，OM模型的精度为  **{Top1Acc=72.15%, Top5@Acc=90.4%}**，当batchsize设为16时模型性能最优，达 8933.6 fps。

| 芯片型号   | BatchSize | 数据集      | 精度            | 性能       |
| --------- | --------- | ----------- | --------------- | --------- |
|Ascend310P3| 1         | ILSVRC2012  | Top1Acc=72.15% Top5@Acc=90.4% | 1938.0 fps |
|Ascend310P3| 4         | ILSVRC2012  | Top1Acc=72.15% Top5@Acc=90.4% | 4694.8 fps |
|Ascend310P3| 8         | ILSVRC2012  | Top1Acc=72.15% Top5@Acc=90.4% | 6739.7 fps |
|Ascend310P3| 16        | ILSVRC2012  | Top1Acc=72.15% Top5@Acc=90.4% | **8933.6 fps** |
|Ascend310P3| 32        | ILSVRC2012  | Top1Acc=72.15% Top5@Acc=90.4% | 8667.4 fps |
|Ascend310P3| 64        | ILSVRC2012  | Top1Acc=72.15% Top5@Acc=90.4% | 5837.8 fps |
