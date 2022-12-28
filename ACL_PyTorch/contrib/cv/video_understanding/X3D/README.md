# X3D 模型推理指导

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

X3D，这是一个高效的视频网络系列，可沿多个网络轴在空间、时间、宽度和深度上逐步扩展微小的 2D 图像分类架构。受机器学习中的特征选择方法的启发，采用了一种简单的逐步网络扩展方法，在每个步骤中扩展单个轴，从而实现对复杂性权衡的良好准确性。

+ 论文  
    [X3D: Expanding Architectures for Efficient Video Recognition](https://openaccess.thecvf.com/content_CVPR_2020/html/Feichtenhofer_X3D_Expanding_Architectures_for_Efficient_Video_Recognition_CVPR_2020_paper.html)    
    Christoph Feichtenhofer; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 203-213  

+ 参考实现：  
    url=https://github.com/facebookresearch/slowfast  
    branch=main  
    commit_id=9839d1318c0ae17bd82c6a121e5640aebc67f126  

## 输入输出数据
+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | image      | RGB_FP32  | NCHW | bs x 3 x 13 x 182 x 182 | 

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | class     |  FLOAT32   | ND          | bs x 400   |


----
# 推理环境

- 该模型推理所需配套的软件如下：

    | 配套      | 版本    | 环境准备指导 |
    | --------- | ------- | ---------- |
    | 固件与驱动 | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN      | 6.0.RC1 | -          |
    | Python    | 3.8.13  | -          |
    
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
    git clone https://github.com/facebookresearch/detectron2
    pip install -v -e detectron2
    
    git clone https://github.com/facebookresearch/SlowFast -b main
    cd SlowFast
    git reset 9839d1318c0ae17bd82c6a121e5640aebc67f126 --hard
    patch -p1 < ../x3d.patch
    python setup.py build develop
    
    cd ..
    ```

## 准备数据集

1. 获取原始数据集  
    本模型使用 Kinetic400 验证集的19761个视频来验证模型精度，可通过[**val_link.list**](https://ai-rank.bj.bcebos.com/Kinetics400/val_link.list)获取Kinetic400验证集的下载链接，下载该文件中的三个压缩包并解压到`Kinetics-400/val`目录下。然后下载标签文件[**val.list**](https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val.list)，将标签文件放置到`Kinetics-400`目录下并重命名为`test.csv`。进行上述操作后，最终数据与标签的目录结构应为：
    ```
    ├── Kinetics-400/
        ├── val/
            ├── abseiling/
            ├── air_drumming/
            ├── ...
            ├── yoga/
            └── zumba/
       └── test.csv
    ```


2. 数据预处理  
    执行前处理脚本将原始数据转换为OM模型输入需要的bin/npy文件。
    ```bash
    python x3d_preprocess.py \
        --cfg SlowFast/configs/Kinetics/X3D_S.yaml \
        DATA.PATH_TO_DATA_DIR Kinetics-400/ \
        DATA.PATH_PREFIX Kinetics-400/val/ \
        X3D_PREPROCESS.DATA_OUTPUT_PATH prep_data/
    ```
    参数说明：
    + --cfg: 模型配置文件路径
    + DATA.PATH_TO_DATA_DIR: 标签文件位路径
    + DATA.PATH_PREFIX: 数据集路径
    + X3D_PREPROCESS.DATA_OUTPUT_PATH: 输出文件路径


## 模型转换

1. PyTroch 模型转 ONNX 模型  
 
    下载[**预训练模型**](https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_s.pyth)到当前目录，可参考命令：
    ```bash
    wget https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_s.pyth
    ```

    然后执行执行以下命令生成 ONNX 模型：
    ```bash
    python x3d_pth2onnx.py \
        --cfg SlowFast/configs/Kinetics/X3D_S.yaml \
        TEST.CHECKPOINT_FILE_PATH  x3d_s.pyth \
        X3D_PTH2ONNX.ONNX_OUTPUT_PATH x3d_s.onnx
    ```
    参数说明：
    + --cfg: 模型配置文件路径
    + TEST.CHECKPOINT_FILE_PATH: 预训练权重文件路径
    + X3D_PTH2ONNX.ONNX_OUTPUT_PATH: 保存生成ONNX模型的路径

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
    bs=1  # 根据需要自行设置batchsize

    
    # 执行 ATC 进行模型转换
    atc --framework=5 \
        --model=x3d_s.onnx \
        --output=x3d_s_bs${bs} \
        --input_format=NCHW \
        --input_shape="image:${bs},3,13,182,182" \
        --log=error \
        --soc_version=Ascend${chip_name} \
        --precision_mode=allow_mix_precision
    ```

   参数说明：
    + --framework: 5代表ONNX模型
    + --model: ONNX模型路径
    + --input_shape: 模型输入数据的shape
    + --input_format: 输入数据的排布格式
    + --output: OM模型路径，无需加后缀
    + --log：日志级别
    + --soc_version: 处理器型号
    + --precision_mode: OM模型的精度模式


## 推理验证

1. 对数据集推理  
    该离线模型使用ais_infer作为推理工具，请参考[**安装文档**](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer#%E4%B8%80%E9%94%AE%E5%AE%89%E8%A3%85)安装推理后端包aclruntime与推理前端包ais_bench。完成安装后，执行以下命令预处理后的数据进行推理。
    ```bash
    python -m ais_bench \
        --model x3d_s_bs${bs}.om \
        --input ./prep_data/ \
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
    python -m ais_bench --model x3d_s_bs${bs}.om --batchsize ${bs} --loop 100
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python x3d_postprocess.py \
        --cfg SlowFast/configs/Kinetics/X3D_S.yaml \
        X3D_POSTPROCESS.ENABLE True \
        X3D_POSTPROCESS.OM_OUTPUT_PATH result_bs${bs}
    ```
    参数说明：
    + --cfg: 模型配置文件路径
    + X3D_POSTPROCESS.ENABLE: 开启后处理
    + X3D_POSTPROCESS.OM_OUTPUT_PATH: 推理结果所在路径
    
    运行成功后，程序会将会打印出模型的精度指标：
    ```
    {"split": "test_final", "top1_acc": "73.75", "top5_acc": "90.25"}
    ```

----
# 性能&精度

在310P设备上，当batchsize设为8时OM模型性能最优，达 **386.8 fps**，此时模型精度为  **{Top1@Acc=73.75%, Top5@Acc=90.25%}**

| 芯片型号   | BatchSize | 数据集       | 精度            | 性能       |
| --------- | --------- | ------------ | --------------- | --------- |
|Ascend310P3| 1         | Kinetics-400 | Top1@Acc=73.75%, Top5@Acc=90.25% | 333.2 fps |
|Ascend310P3| 4         | Kinetics-400 |  | 381.6 fps |
|Ascend310P3| 8         | Kinetics-400 | Top1@Acc=73.75%, Top5@Acc=90.25% | **386.8 fps** |
|Ascend310P3| 16        | Kinetics-400 |  | 365.2 fps |
|Ascend310P3| 32        | Kinetics-400 | Top1@Acc=73.75%, Top5@Acc=90.25% | 358.9 fps |
|Ascend310P3| 64        | Kinetics-400 |  | 354.9 fps |
