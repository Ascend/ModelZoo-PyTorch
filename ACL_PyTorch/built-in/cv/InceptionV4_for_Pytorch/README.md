# InceptionV4 模型推理指导

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

InceptionV4中基本的Inception module还是沿袭了Inception v2/v3的结构，只是结构看起来更加简洁统一，并且使用更多的Inception module，实验效果也更好。

+ 论文
    [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)  
    Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi  

+ 参考实现：  
    url=https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionv4.py  
    branch=master  
    commit_id=8aae3d8f1135b6b13fed79c1d431e3449fdbf6e0  
    model_name=InceptionV4

## 输入输出数据
+ 模型输入  
    | 输入数据 | 数据类型  | 数据排布格式 | 大小               |
    | ------- | -------- | ----------- | ------------------ |
    | image   | RGB_FP32 | NCHW        | bs x 3 x 299 x 299 | 

+ 模型输出  
    | 输出数据     | 数据类型    | 数据排布格式 | 大小        |
    | ----------- | ---------- | ----------- | ----------- |
    | class       |  FLOAT32   | ND          | bs x 1001   |


----
# 推理环境

- 该模型推理所需配套的软件如下：

    | 配套      | 版本    | 环境准备指导 |
    | --------- | ------- | ---------- |
    | 固件与驱动 | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN      | 6.0.RC1 | -          |
    | Python    | 3.7.5   | -          |
    
    说明：请根据推理卡型号与 CANN 版本选择相匹配的固件与驱动版本。


----
# 快速上手

## 获取源码

1. 安装推理过程所需的依赖
    ```bash
    pip install -r requirements.txt
    ```
2. 获取开源仓源码
    ```bash
    git clone https://github.com/Cadene/pretrained-models.pytorch.git
    cd pretrained-models.pytorch
    git checkout master
    git reset --hard 8aae3d8f1135b6b13fed79c1d431e3449fdbf6e0
    cd ..
    ```
3. 更改源码
    ```bash
    patch ./pretrained-models.pytorch/pretrainedmodels/models/inceptionv4.py inceptionv4.patch
    ```

## 准备数据集

1. 获取原始数据集  
    本模型推理项目使用 ILSVRC2012 数据集验证模型精度，请在 [ImageNet官网](https:image-net.org) 自行下载，并按照以下的目录结构存放图片与标签文件。   
    ```
    ├── imageNet/
        ├── val/
            ├── ILSVRC2012_val_00000001.JPEG
            ├── ILSVRC2012_val_00000002.JPEG
            ├── ...
            ├── ILSVRC2012_val_00050000.JPEG
        ├── val_label.txt
    ```


2. 数据预处理  
    执行前处理脚本将原始数据转换为OM模型输入需要的bin/npy文件。
    ```bash
    python inceptionv4_preprocess.py --src_path /opt/npu/imageNet/val --save_path ./prep_dataset
    ```
    参数说明：
    + --src_path: 测试图片所在的目录路径
    + --save_path: 存放生成的bin文件的目录路径
    
    运行成功后，每张原始图片都会对应生成一个bin文件存放于 ./prep_dataset 目录下，总计50000个bin文件。


## 模型转换

1. PyTroch 模型转 ONNX 模型  
 
    下载PyTorch官方提供的[ **预训练模型** ](http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth) 到当前目录，可参考命令：
    ```bash
    wget http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth --no-check-certificate
    ```

    然后执行执行以下命令生成 ONNX 模型：
    ```bash
    python inceptionv4_pth2onnx.py --ckpt ./inceptionv4-8e4777a0.pth --onnx ./inceptionv4.onnx
    ```
    参数说明：
    + --ckpt: 预训练权重文件的路径。若不指定，则会通过在线方式获取。
    + --onnx: 生成ONNX模型的保存路径

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
    bs=8  # 根据需要自行设置batchsize

    
    # 执行 ATC 进行模型转换
    atc --framework=5 \
        --model=inceptionv4.onnx \
        --output=inceptionv4_bs${bs} \
        --input_format=NCHW \
        --input_shape="image:${bs},3,299,299" \
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


## 推理验证

1. 对数据集推理  
    安装ais_bench推理工具。请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。完成安装后，执行以下命令预处理后的数据进行推理。
    ```bash
    python -m ais_bench \
        --model inceptionv4_bs${bs}.om \
        --input ./prep_dataset/ \
        --output ./ \
        --output_dirname ./result_bs${bs}/ \
        --outfmt TXT \
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
    python -m ais_bench --model inceptionv4_bs${bs}.om --batchsize ${bs} --loop 100
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python inceptionv4_postprocess.py \
        --infer_results ./result_bs${bs}/ \
        --anno_file /opt/npu/imageNet/val_label.txt \
        --metrics_json metrics.json
    ```
    参数说明：
    + --infer_results: 存放推理结果的目录路径
    + --anno_file: 标签文件路径
    + --metrics_json: 指定一个json文件用于保存指标信息。
    
    运行成功后，程序会将各top1~top5的正确率记录在 metrics.json 文件中，可执行以下命令查看：
    ```bash
    python -m json.tool metrics.json
    ```

----
# 性能&精度

在310P设备上，OM模型的精度为  **{Top1@Acc=79.99%, Top5@Acc=94.86%}**，当batchsize设为4时模型性能最优，达 1498.5 fps。

| 芯片型号   | BatchSize | 数据集      | 精度            | 性能       |
| --------- | --------- | ----------- | --------------- | --------- |
|Ascend310P3| 1         | ILSVRC2012  | Top1@Acc=79.99% Top5@Acc=94.86% | 761.4 fps |
|Ascend310P3| 4         | ILSVRC2012  | Top1@Acc=79.99% Top5@Acc=94.86% | 1498.5 fps |
|Ascend310P3| 8         | ILSVRC2012  | Top1@Acc=79.99% Top5@Acc=94.86% | 1315.3 fps |
|Ascend310P3| 16        | ILSVRC2012  | Top1@Acc=79.99% Top5@Acc=94.86% | 1181.3 fps |
|Ascend310P3| 32        | ILSVRC2012  | Top1@Acc=79.99% Top5@Acc=94.86% | 1163.6 fps |
|Ascend310P3| 64        | ILSVRC2012  | Top1@Acc=79.99% Top5@Acc=94.86% | 1168.9 fps |


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md