#  Fsaf 模型推理指导

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

FSAF 是 CVPR2019发表的一种无锚定方法。实际上它等价于基于锚的方法，在每个 FPN 级别的每个特征映射位置只有一个锚。我们就是这样实施的。只有没有锚的分支被释放，因为它与当前框架的兼容性更好，计算预算更少。

+ 论文  
    [Fsaf论文](https://arxiv.org/pdf/1903.00621.pdf)
    Chenchen Zhu Yihui He Marios Savvides

+ 参考实现：
    ```  
    https://github.com/open-mmlab/mmdetection
    branch:master  
    commit_id:604bfe9618533949c74002a4e54f972e57ad0a7a
    ```
## 输入输出数据
+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | actual_input_1 | FLOAT32 | NCHW | bs x 3 x 800 x 1216 | 

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output1      |  FLOAT32   | ND          | bs x 100 x 5        |
    | output2      |  FLOAT32   | ND          | bs x 100         |

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

## 安装

- 安装推理过程所需的依赖
    ```bash
    pip3 install -r requirements.txt
    ```
- 获取源码
    ```
    git clone https://github.com/open-mmlab/mmcv -b master 
    cd mmcv
    git reset --hard 04daea425bcb0a104d8b4acbbc16bd31304cf168
    MMCV_WITH_OPS=1 pip3.7 install -e .
    cd ..
    git clone https://github.com/open-mmlab/mmdetection -b master
    cd mmdetection
    git reset --hard 604bfe9618533949c74002a4e54f972e57ad0a7a
    patch -p1 < ../fsaf.diff
    pip3 install -r requirements/build.txt
    python3 setup.py develop
    cd ..
    ```
## 准备数据集

1. 获取原始数据集  
    本模型支持coco的验证集。请用户需自行获取coco数据集，上传数据集到服务器任意目录并解压（如：/root/datasets/）。本模型将使用到val2017验证集及annotations中的instances_val2017.json，获取后的数据集结构如下
    ```
    root
    ├── datasets
    │   ├── coco
    │   │   ├── annotations
    │   │   ├── val2017
    │   │   │   ├── 0000000001.jpg
    │   │   │   ├── 0000000002.jpg
    │   │   │   ├── ·······
    ```


2. 数据预处理  
    执行前处理脚本将原始数据转换为OM模型输入需要的bin/npy文件。
    ```
    python3 Fsaf_preprocess.py \
        --image_src_path=/root/datasets/coco/val2017 \
        --bin_file_path=val2017_bin \
        --model_input_height=800 \
        --model_input_width=1216
    ```

    参数说明：
    + --image_src_path: 原始数据集路径
    + --bin_file_path: 转化后的bin文件路径
    + --model_input_height：输入图片的高
    + --model_input_width：输入图片的宽
    
    生成数据集图片路径信息
    ```
    python3 get_info.py --file_type jpg --file_path ${datasets_path}/coco/val2017 --info_name fsaf_jpeg.info
    ```
    参数说明：
    + --file_type: 数据集文件格式
    + --file_path: 数据集路径
    + --info_name：输出信息路径

## 模型转换


1. 下载pth权重文件  
[FSAF预训练pth权重文件](https://github.com/open-mmlab/mmdetection/tree/master/configs/fsaf)  
进入下载页面链接 此页面下载box AP=37.4的权重pth文件

    然后执行执行以下命令生成 ONNX 模型：
    ```bash
    python3 ./mmdetection/tools/deployment/pytorch2onnx.py ./mmdetection/configs/fsaf/fsaf_r50_fpn_1x_coco.py ./fsaf_r50_fpn_1x_coco-94ccc51f.pth --output-file fsaf.onnx --input-img ./mmdetection/demo/demo.jpg --shape 800 1216
    ```
    参数说明：
    + --output-file : 权重文件路径
    + --input-img : 输出onnx的文件路径
    + --shape ：输出onnx的可输入数据量


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
    bs=1
    
    # 执行 ATC 进行模型转换
    atc --model=./fsaf.onnx \
        --framework=5 \
        --output=./fsaf_bs1  \
        --input_format=NCHW \
        --input_shape="input:1,3,800,1216" \
        --log=error \
        --soc_version=Ascend${chip_name} \
        --out_nodes="dets;labels"
    ```

   参数说明：
    + --framework: 5代表ONNX模型
    + --model: ONNX模型路径
    + --input_shape: 模型输入数据的shape
    + --input_format: 输入数据的排布格式
    + --output: OM模型路径，无需加后缀
    + --log：日志级别
    + --soc_version: 处理器型号
    + --out_nodes: 输出节点名


## 推理验证

1. 对数据集推理  
    该离线模型使用ais_infer作为推理工具，请参考[**安装文档**](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer#%E4%B8%80%E9%94%AE%E5%AE%89%E8%A3%85)安装推理后端包aclruntime与推理前端包ais_bench。完成安装后，执行以下命令预处理后的数据进行推理。
    ```bash
    python3 -m ais_bench \
        --model ./fsaf_bs1.om \
        --input ./prep_dataset/ \ 
        --output ./ \
        --output_dirname ./result/ \
        --outfmt BIN \
        --batchsize 1
    ```
    参数说明：
    + --model OM模型路径
    + --input 存放预处理后数据的目录路径
    + --output 用于存放推理结果的父目录路径
    + --output_dirname 用于存放推理结果的子目录名，位于--output指定的目录下
    + --outfmt 推理结果文件的保存格式
    + --batchsize 模型每次输入bin文件的数量,本例中为1。


2. 性能验证  
    对于性能的测试，需要注意以下三点：
    + 测试前，请通过`npu-smi info`命令查看NPU设备状态，请务必在NPU设备空闲的状态下进行性能测试。
    + 为了避免测试过程因持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps，反映模型在单位时间（1秒）内处理的样本数。
    ```bash
    python3 -m ais_bench --model ./model/fsaf_bs1.om --batchsize ${bs}
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python3 Fsaf_postprocess.py --bin_data_path=./result/dumpOutput_device0/ --test_annotation=fsaf_jpeg.info  --net_out_num=3 --net_input_height=800 --net_input_width=1216 --annotations_path=/root/datasets/coco/annotations/instances_val2017.json
    ```
    参数说明：
    + --bin_data_path: 存放推理结果的目录路径
    + --test_annotation: 图片信息文件
    + --net_out_num: 输出节点数
    + --net_input_height: 网络输入高度
    + --net_input_width: 网络输入宽度
    + --annotations_path: 标签文件路口
    控制台输出如下信息

    ```
    Evaluating bbox...
    Loading and preparing results...
    DONE (t=1.98s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=52.96s).
    Accumulating evaluation results...
    DONE (t=18.60s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.371
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.565
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.395
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.200
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.405
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.499
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.544
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.544
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.544
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.336
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.588
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.705
    ```
    


----
# 性能&精度

在310P设备上，OM模型的精度为  **{map=0.371}**，当batchsize设为1时模型性能最优，达 20.2 fps。

| 芯片型号   | BatchSize | 数据集      | 精度            | 性能       |
| --------- | --------- | ----------- | --------------- | --------- |
|Ascend310P3| 1         | COCO2017  | map=0.371| 20.2 fps |
|Ascend310P3| 4         | COCO2017  | map=0.371| 9.0 fps |
|Ascend310P3| 8         | COCO2017  | map=0.371| 4.9 fps |
|Ascend310P3| 16        | COCO2017  | map=0.371| 2.4 fps |
|Ascend310P3| 32        | COCO2017  | map=0.371| 1.2 fps |
|Ascend310P3| 64        | COCO2017  | map=0.371| 0.5 fps |