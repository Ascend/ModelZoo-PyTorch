# LResNet100E-IR Onnx模型端到端推理指导

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

在本文中，作者首先介绍了一种附加角裕度损失（ArcFace），它不仅具有清晰的几何解释还显著增强了辨别力。由于ArcFace容易受到大量标签的影响噪声，我们进一步提出了子中心ArcFace，其中每个类包含K个子中心，训练样本只需要接近K个正子中心中的任何一个。副中心ArcFace鼓励一个占主导地位的子类，其中包含大多数干净的面部和包括硬面部或噪声面部的非优势子类。基于这种自行隔离，我们提高了性能通过在巨大的真实世界噪音下自动净化原始网页。除了鉴别特征嵌入，我们还探索逆问题，将特征向量映射到人脸图像。无需培训任何额外的发生器或鉴别器预训练的ArcFace模型只能通过以下方式为训练数据内外的受试者生成保持身份的人脸图像使用网络梯度和批量归一化（BN）先验。大量实验表明，ArcFace可以增强识别特征嵌入以及增强生成人脸合成。

+ 论文  
    [LResNet100E-IR论文](https://arxiv.org/pdf/1801.07698.pdf)  
    Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Kotsia, and Stefanos Zafeiriou

+ 参考实现：  
    https://github.com/TreB1eN/InsightFace_Pytorch

## 输入输出数据
+ 模型输入    
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | image | FLOAT32 | NCHW | batch_size x 3 x 112 x 112 | 

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output1      |  FLOAT32   | ND          | batch_size x 512        |


----
# 推理环境

- 该模型推理所需配套的软件如下：

    | 配套      | 版本    | 环境准备指导 |
    | --------- | ------- | ---------- |
    | 固件与驱动 | 1.0.20.alpha | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN      | 7.0.RC1 | -          |
    | Python    | 3.7.5   | -          |
    
    说明：请根据推理卡型号与 CANN 版本选择相匹配的固件与驱动版本。


----
# 快速上手

## 安装

- 安装推理过程所需的依赖
    ```bash
    pip3 install -r requirements.txt
    ```
- 获取源码
    ```bash
    cd LResNet100E-IR

    git clone https://github.com/TreB1eN/InsightFace_Pytorch.git ./LResNet
    cd LResNet
    patch -p1 < ../LResNet.patch
    rm -rf ./work_space/* 
    mkdir ./work_space/history && mkdir ./work_space/log && mkdir ./work_space/models && mkdir ./work_space/save
    cd ..
    ```
## 准备数据集
获取LFW数据集，放在工作目录的data目录下


1. 获取原始数据集  
    OBS： [lfw.bin](obs://l-resnet100e-ir/infer/lfw.bin) 云盘： [lfw.bin](https://drive.google.com/file/d/1mRB0A8f0b5GhH7w0vNMGdPjSWF-VJJLY/view?usp=sharing) 
 
    ```bash
    mkdir data
    mv lfw.bin ./data
    ```


2. 数据预处理  
    执行前处理脚本将原始数据转换为OM模型输入需要的bin/npy文件。
    ```bash
    python3 LResNet_preprocess.py --file_type jpg --data_path ./data/lfw.bin --info_path ./data/lfw --width 112  --height 112
    ```
    其中"file_type"表示生成数据 bin 文件和 target 文件模式，"data_path"表示原始数据集路径，"info_path"表示数据集保存路径，"width,height"表示数据图片宽高


## 模型转换

1. PyTroch 模型转 ONNX 模型  

    获取模型权重，并放在工作目录的model文件夹下
    OBS： [model_ir_se100.pth](obs://l-resnet100e-ir/infer/model_ir_se100.pth)  云盘：[model_ir_se100.pth](https://drive.google.com/file/d/1rbStth01wP20qFpot06Cy6tiIXEEL8ju/view?usp=sharing)

    ```bash
    mkdir model
    mv model_ir_se100.pth ./model/
    ```
 
    然后执行执行以下命令生成 ONNX 模型：
    ```
    python3 LResNet_pth2onnx.py --source ./model/model_ir_se100.pth --target ./model/model_ir_se100_bs1.onnx --batchsize 1
    python3 -m onnxsim --input-shape="1,3,112,112" ./model/model_ir_se100_bs1.onnx ./model/model_ir_se100_bs1_sim.onnx

    ```
    参数说明：
     + --source: 预训练权重文件的路径。若不指定，则会通过在线方式获取。
     + --target: 生成ONNX模型的保存路径
     + --batchsize: 模型batch

2. ONNX 模型转 OM 模型  

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
    batch_size=1  # 根据需要自行设置 

    
    # 执行 ATC 进行模型转换
    atc --framework=5 --model=./model/model_ir_se100_bs${batch_size}_sim.onnx --output=model/model_ir_se100_bs${batch_size} --input_format=NCHW --input_shape="image:${batch_size},3,112,112" --log=debug --soc_version=Ascend${chip_name}
    ```

   参数说明：
    + --framework: 5代表ONNX模型
    + --model: ONNX模型路径
    + --input_shape: 模型输入数据的shape
    + --input_format: 输入数据的排布格式
    + --output: OM模型路径，无需加后缀
    + --log：日志级别
    + --soc_version: 处理器型号

## 推理验证

1. 对数据集推理  
    安装ais_bench推理工具。请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。完成安装后，执行以下命令预处理后的数据进行推理。
    ```bash
    mkdir result
    python3 -m ais_bench \
        --model model/model_ir_se100_bs${batch_size}.om \
        --input ./data/lfw/ \ 
        --output ./result/ \
        --outfmt TXT \
        --batchsize ${batch_size}
    ```
    参数说明：
    + --model OM模型路径
    + --input 存放预处理后数据的目录路径
    + --output 用于存放推理结果的父目录路径
    + --outfmt 推理结果文件的保存格式
    + --batchsize 模型每次输入bin文件的数量


2. 性能验证  
    对于性能的测试，需要注意以下三点：
    + 测试前，请通过`npu-smi info`命令查看NPU设备状态，请务必在NPU设备空闲的状态下进行性能测试。
    + 为了避免测试过程因持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps，反映模型在单位时间（1秒）内处理的样本数。
    ```bash
    python3 -m ais_bench --model model/model_ir_se100_bs${batch_size}.om --batchsize ${batch_size} --loop 100
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python3 LResNet_postprocess.py \
        --result ./result/ \
        --data_path ./data/lfw_list.npy \
    ```
    参数说明：
    + --result: 存放推理结果的目录路径
    + --data_path: 原始数据集路径
    
    运行成功后，控制台输出如下信息：
    ```
    accuracy: 0.9976666666666667
    best_thresholds: 1.4140000000000001
    ```



----
# 性能&精度

在310P设备上，OM模型的精度为  **Acc=99.76%**，当batchsize设为16时模型性能最优，达 **1432fps**。

| 芯片型号   | BatchSize | 数据集      | 精度            | 性能       |
| --------- | --------- | ----------- | --------------- | --------- |
|Ascend310P3| 1         | IFW | 0.9976 | 566 fps |
|Ascend310P3| 4         | IFW | 0.9976 | 1136 fps |
|Ascend310P3| 8         | IFW | 0.9976 | 1373 fps |
|Ascend310P3| 16        | IFW | 0.9976 | 1432 fps |
|Ascend310P3| 32        | IFW | 0.9976 | 1384 fps |
|Ascend310P3| 64        | IFW | 0.9976 | 1305 fps |
