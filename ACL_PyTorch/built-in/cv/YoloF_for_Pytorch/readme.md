# YoloF 模型推理指导

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


YOLOF引入了一种解决该优化问题的替代方案而无需使用复杂的特征金字塔，只需要使用单层特征图。基于这个简单高效的方案，作者设计了You Only Look One-level Feature（YOLOF）检测框架。在该框架中，有两个核心组件，分别是膨胀编码器（Dilated Encoder）和均衡匹配策略（Uniform Matching），它们带来了巨大的性能提升。COCO数据集上的实验证明了YOLOF的有效性，它获得了和有特征金字塔版本相当的结果但是速度快了2.5倍。此外，在没有Transformer层的前提下，YOLOF可以和同样使用单层特征图的DETR媲美且训练轮次少了7倍。


+ 论文  
    [You Only Look One-level Feature](https://arxiv.org/abs/2103.09460)  
    Qiang Chen, Yingming Wang, Tong Yang, Xiangyu Zhang, Jian Cheng, Jian Sun

+ 参考实现：  
    https://github.com/open-mmlab/mmdetection.git

## 输入输出数据
+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | input | FLOAT32 | NCHW | batch_size x 3 x 640 x 640 | 

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output1      |  FLOAT32   | ND          | batchsize x num_dets x 5        |
    | output1      |  INT32   | ND          | batchsize x num_dets        |

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

- 获取源码
    ```bash
   git clone https://github.com/open-mmlab/mmdetection.git
   cd mmdetection
   git reset 3e2693151add9b5d6db99b944da020cba837266b --hard
   pip3 install -v -e .
   mmdetection_path=$(pwd)
   git apply ../mmdetction.patch
   cd ..
   git clone https://github.com/open-mmlab/mmdeploy.git
   cd mmdeploy
   git reset 0cd44a6799ec168f885b4ef5b776fb135740487d --hard
   pip3 install -v -e .
   mmdeploy_path=$(pwd)
   git apply ../mmdeploy.patch
    ```

    ```bash
    pip3 install -r requirements.txt
    ```

1. 获取原始数据集  
    本模型支持coco2017验证集。用户需自行获取数据集，将annotations文件和val2017文件夹解压并上传数据集到源码包路径下的dataset文件夹下。目录结构如下：

      ```
      ├── annotations
      └── val2017 
      ```


2. 数据预处理  
    执行前处理脚本将原始数据转换为OM模型输入需要的bin/npy文件。
    ```bash
    python3 YOLOF_preprocess.py --image_src_path ${datasets_path}/val2017
    ```
    其中"datasets_path"表示处理前原数据集的地址，运行完成后，将在根目录下生成'val2017_bin'及'val2017_bin_meta'两个后处理文件夹
    
    运行后，将会得到如下形式的文件夹：

    ```
    ├── val2017_bin
    │    ├──000000000139.bin
    │    ├──......     	 
    ├── val2017_bin_meta
    │    ├──000000000139.pk
    │    ├──......     	 
    ```


## 模型转换

1. PyTroch 模型转 ONNX 模型  

    下载YOLOF对应的[weights](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolof)，名称为yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth,放到mmdeploy_path目录下
 
    然后执行执行以下命令生成 ONNX 模型：
    ```
    cd mmdeploy
    python3 tools/deploy.py \
        configs/mmdet/detection/detection_onnxruntime_dynamic.py \
        ${mmdetection_path}/configs/yolof/yolof_r50_c5_8x8_1x_coco.py  \
        yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth \
        ${mmdetection_path}/demo/demo.jpg \
        --work-dir work_dir
    cd ..
    ```

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
    atc --model=./mmdeploy/work_dir/end2end.onnx  \
        --framework=5 \
        --output=./ais/yolof_bs${batch_size} \
        --input_format=NCHW \
        --input_shape="input:$batch_size,3,640,640" \
        --log=error \
        --op_precision_mode=op_precision.ini \
        --soc_version=Ascend${chip_name} \
        --insert_op_conf=aipp.conf \
        --enable_small_channel=1
    ```

   参数说明：
    + --framework: 5代表ONNX模型
    + --model: ONNX模型路径
    + --input_shape: 模型输入数据的shape
    + --input_format: 输入数据的排布格式
    + --output: OM模型路径，无需加后缀
    + --log：日志级别
    + --soc_version: 处理器型号
    + --optypelist_for_implmode: 列举算子optype的列表
    + --enable_small_channel: 是否使能small channel的优化
    + --op_precision_mode: 算子精度模式
    + --insert_op_conf: 插入算子描述


    
## 推理验证

1. 对数据集推理  
    安装ais_bench推理工具。请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。完成安装后，执行以下命令预处理后的数据进行推理。
    ```bash
    python3 -m ais_bench \
        --model yolof_bs${batch_size}.om \
        --input val2017_bin \ 
        --output ./ \
        --output_dirname result
        --outfmt BIN\
        --batchsize ${batch_size}
    ```
    参数说明：
    + --model OM模型路径
    + --input 存放预处理后数据的目录路径
    + --output 用于存放推理结果的父目录路径
    + --outfmt 推理结果文件的保存格式
    + --batchsize 模型每次输入bin文件的数量
    + --output_dirname 结果文件夹名称

2. 性能验证  
    对于性能的测试，需要注意以下三点：
    + 测试前，请通过`npu-smi info`命令查看NPU设备状态，请务必在NPU设备空闲的状态下进行性能测试。
    + 为了避免测试过程因持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps，反映模型在单位时间（1秒）内处理的样本数。
    ```bash
    python3 -m ais_bench --model yolof_bs${batch_size}.om --batchsize ${batch_size}
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的精度：
    (1)生成数据集信息数据
    ```bash
    python3 gen_dataset_info.py \
        --image_src_path ${datasets_path} \
        --config_path ${mmdetection_path}/configs/yolof/yolof_r50_c5_8x8_1x_coco.py  \
        --bin_path val2017_bin  --meta_path val2017_bin_meta  \
        --info_name yolof.info  --info_meta_name yolof_meta.info  \
        --width 640 --height 640
    ```
    参数说明：
    + --image_src_path: 存放推理结果的目录路径
    + --config_path: 标签文件路径
    + --bin_path val2017_bin: 后处理bin结果文件夹
    + --meta_pathe: 后处理pk结果文件夹
    + --info_name: 数据集信息文件路径
    + --info_meta_name: 数据集meta信息文件路径
    + --width: 模型输入宽度。
    + --height: 模型输入高度

    (2)生成精度数据
    ```bash
    python3 YOLOF_postprocess.py --dataset_path ${datasets_path} --model_config ${mmdetection_path}/configs/yolof/yolof_r50_c5_8x8_1x_coco.py \
    ```

    参数说明：
    + --dataset_path: 数据集路径
    + --model_config: 模型配置文件路径


    运行成功后，程序会将正确率记录在 results.txt 文件中，可执行以下命令查看：
    ```
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.303
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.474
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.320
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.088
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.330
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.520
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.429
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.429
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.429
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.127
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.507
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.701

    acc:0.303
    ```


----
# 性能&精度

在310P设备上，OM模型的精度为  **{Top1@Acc=0.303}**，当batchsize设为1时模型性能最优，达 266.8 fps。

| 芯片型号   | BatchSize | 数据集      | 精度            | 性能       |
| --------- | --------- | ----------- | --------------- | --------- |
|Ascend310P3| 1         | COCO  | Map=0.303 | 285.615 fps |
|Ascend310P3| 4         | COCO  | Map=0.303 | 276.412 fps |
|Ascend310P3| 8         | COCO  | Map=0.303 | 264.419 fps |
|Ascend310P3| 16        | COCO  | Map=0.303 | 267.215 fps |
|Ascend310P3| 32        | COCO  | Map=0.303 | 268.421 fps |
|Ascend310P3| 64        | COCO  | Map=0.303 | 233.776 fps |