# SRFlow 模型推理指导

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

SRFlow是一种基于归一化流的超分辨率方法，具备比GAN更强的脑补能力，能够基于低分辨率输入学习输出的条件分布。

+ 论文  
    [SRFlow: Learning the Super-Resolution Space with Normalizing Flow](https://arxiv.org/abs/2006.14200)  
    Andreas Lugmayr, Martin Danelljan, Luc Van Gool, Radu Timofte  

+ 参考实现：  
    url = https://github.com/andreas128/SRFlow  
    branch = master  
    commit_id = 8d91d81a2aec17e7739c5822f3a5462c908744f8  
    model_name = SRFlow  

## 输入输出数据
+ 模型输入  
    | input-name  | data-type | data-format |input-shape |
    | ----------- | --------- | ----------- | ---------- |
    | input-image | RGB_FP32  | NCHW        | batchsize x 3 x 256 x 256 | 

+ 模型输出  
    | output-name  |  data-type | data-format |output-shape |
    | ------------ | ---------- | ----------- | ----------- |
    | output-image |  RGB_FP32  | NCHW        | batchsize x 3 x 2048 x 2048   |

    说明：目前该模型只支持转出 batchszie 为 1 的 OM 模型。
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
    git clone https://github.com/andreas128/SRFlow -b master
    cd SRFlow
    git reset 8d91d81a2aec17e7739c5822f3a5462c908744f8 --hard
    patch -p1 < ../srflow.patch
    cd ..
    ```

## 准备数据集

1. 获取原始数据集  
    本模型使用[DIV2K数据集](https://data.vision.ee.ethz.ch/cvl/DIV2K/)来验证模型精度，可参考以下命令下载并解压原始数据集：
    ```bash
    wget http://data.vision.ee.ethz.ch/alugmayr/SRFlow/datasets.zip
    unzip datasets.zip
    ```
    该模型需要的数据及目录结构如下：
    ```
    ├── datasets
        ├── div2k-validation-modcrop8-gt
        └── div2k-validation-modcrop8-x8
    ```

2. 数据预处理  
    执行前处理脚本将原始数据转换为OM模型输入需要的bin文件。
    ```bash
    python srflow_preprocess.py -s ./datasets/div2k-validation-modcrop8-x8 -o ./prep_data
    ```
    参数说明：
    + -s/--source: 原始数据路径
    + -o/--output: 保存输出bin文件的目录路径
    
    运行成功后，每张测试图片都会对应生成一个bin文件存放于`./prep_data/bin`目录下，总数为100.


## 模型转换

1. PyTroch 模型转 ONNX 模型  
 
    下载[**预训练模型**](http://data.vision.ee.ethz.ch/alugmayr/SRFlow/pretrained_models.zip)到当前目录并解压，可参考命令：
    ```bash
    wget http://data.vision.ee.ethz.ch/alugmayr/SRFlow/pretrained_models.zip
    unzip pretrained_models.zip
    ```

    然后执行执行以下命令生成 ONNX 模型：
    ```bash
    python srflow_pth2onnx.py  --pth ./pretrained_models/SRFlow_DF2K_8X.pth --onnx srflow_df2k_x8.onnx
    ```
    参数说明：
    + --pth: 预训练权重文件路径
    + --onnx: ONNX模型的保存路径

2. ONNX 模型优化  
    使用 onnx-simplifier 简化 ONNX 模型：
    ```bash
    python -m onnxsim ./srflow_df2k_x8.onnx ./srflow_df2k_x8_sim.onnx
    ```
    以上的命令中，`./srflow_df2k_x8.onnx`为原始ONXN模型，`./srflow_df2k_x8_sim.onnx`为优化后的ONNX模型。

3. ONNX 模型转 OM 模型  

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
    bs=1  # 根据需要自行设置batchsize

    # 执行 ATC 进行模型转换
    atc --framework=5 \
        --model=./srflow_df2k_x8_sim.onnx \
        --output=./srflow_df2k_x8_bs${bs} \
        --input_format=NCHW \
        --input_shape="input-image:${bs},3,256,256" \
        --log=error \
        --soc_version=Ascend${chip_name} \
        --fusion_switch_file=./srflow_fusion_switch.cfg
    ```

   参数说明：
    + --framework: 5代表ONNX模型
    + --model: ONNX模型路径
    + --input_shape: 模型输入数据的shape
    + --input_format: 输入数据的排布格式
    + --output: OM模型路径，无需加后缀
    + --log：日志级别
    + --soc_version: 处理器型号
    + --fusion_switch_file: OM融合算子配置文件


## 推理验证

1. 对数据集推理  
    安装ais_bench推理工具。  
	请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。  
	完成安装后，执行以下命令预处理后的数据进行推理。
    ```bash
    python -m ais_bench \
        --model ./srflow_df2k_x8_bs${bs}.om \
        --input ./prep_data/bin/ \
        --output ./ \
        --output_dirname result_bs${bs} \
        --batchsize ${bs}
    ```
    参数说明：
    + --model OM模型路径
    + --input 存放预处理后数据的目录路径
    + --output 用于存放推理结果的父目录路径
    + --output_dirname 用于存放推理结果的子目录名，位于--output指定的目录下
    + --batchsize 模型每次输入bin文件的数量


2. 性能验证  
    对于性能的测试，需要注意以下三点：
    + 测试前，请通过`npu-smi info`命令查看NPU设备状态，请务必在NPU设备空闲的状态下进行性能测试。
    + 为了避免测试过程因持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps，反映模型在单位时间（1秒）内处理的样本数。
    ```bash
    python -m ais_bench --model ./srflow_df2k_x8_bs${bs}.om --batchsize ${bs} --loop 100
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python srflow_postprocess.py \
        --hr ./datasets/div2k-validation-modcrop8-gt/ \
        --binres ./result_bs${bs} \
        --save ./result_bs${bs}_save
    ```
    参数说明：
    + --hr: 高分辨率图片所在路径
    + --binres: 推理结果所在路径
    + --save: 后处理生成图片的保存路径
    

----
# 性能&精度

在310P设备上，该模型目前仅支持的batchsize为1，此时OM模型的性能与精度如下表：

| 芯片型号   | BatchSize | 数据集 | 精度            | 性能       |
| --------- | --------- | ------ | --------------- | --------- |
|Ascend310P3| 1         | DIV2K  | avg_psnr=23.017 | 0.737 fps |
