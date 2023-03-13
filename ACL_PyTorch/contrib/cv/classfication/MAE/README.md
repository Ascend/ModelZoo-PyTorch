# MAE 模型推理指导

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

MAE的方法非常简单，随机MASK住图片里的一些块，然后再去重构这些被MASK住的像素。这整个思想也来自 BERT 的带掩码的语言模型，但不一样的是这一个词(patches) 它就是一个 image 的一个块，然后它预测的是你这块里面的所有的像素。

+ 论文  
    [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/pdf/2111.06377.pdf)  
    Kaiming He∗,† Xinlei Chen∗ Saining Xie Yanghao Li Piotr Dollar Ross Girshick

+ 参考实现：  
    https://github.com/facebookresearch/mae

## 输入输出数据
+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | image | FLOAT32 | NCHW | batch_size x 3 x 224 x 224 | 

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output1      |  FLOAT32   | ND          | batch_size x 1000        |


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
    ```bash
    git clone https://github.com/facebookresearch/mae.git
    cd mae
    git reset --hard be47fef7a727943547afb0c670cf1b26034c3c89
    cd ..
    ```
## 准备数据集

1. 获取原始数据集  
    本模型推理项目使用 ILSVRC2012 数据集验证模型精度，请在 [ImageNet官网](https://gitee.com/link?target=http%3A%2F%2Fimage-net.org%2F) 自行下载，并按照以下的目录结构存放图片与标签文件。   
    ```
    ILSVRC2012
    ├── val_label.txt
    ├── images
    │   ├── ILSVRC2012_val_00000001.jpeg
    │   ├── ILSVRC2012_val_00000002.jpeg
    │   .....
    │   ├── ILSVRC2012_val_00050000.jpeg
    ```


2. 数据预处理  
    执行前处理脚本将原始数据转换为OM模型输入需要的bin/npy文件。
    ```bash
    python3 MAE_preprocess.py --image-path /opt/npu/imageNet/val --prep-image ./prep_dataset_batch_size1/ --batch-size 1
    ```
    其中"image-path"表示处理前原数据集的地址，"prep-image"表示生成数据集的文件夹名称(将在文件夹名称后会自动标识对应batchsize，"batch-size"表示生成数据集对应的batchsize（建议使用默认值1,即可支持所有batchsize的推理）

    
    运行后，将会得到如下形式的文件夹：

    ```
    ├── prep_dataset_batch_size1
    │    ├──input_00000.bin
    │    ├──......     	 
    ```


## 模型转换

1. PyTroch 模型转 ONNX 模型  

    使用开源仓提供的mae_finetuned_vit_base.pth
    
    链接：https://pan.baidu.com/s/1FwIK2db5nojOT7YC6rI1Hg 
    提取码：1234 
 
    然后执行执行以下命令生成 ONNX 模型：
    ```
    python3 MAE_pth2onnx.py --source "./mae_finetuned_vit_base.pth" --target "./mae_dynamicbatch_size.onnx"
    ```
    参数说明：
     + --source: 预训练权重文件的路径。若不指定，则会通过在线方式获取。
     + --target: 生成ONNX模型的保存路径

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
    atc --model=./mae_dynamicbatch_size.onnx \
        --framework=5 \
        --output=mae_batch_size${batch_size} \
        --input_format=NCHW \
        --input_shape="image:${batch_size},3,224,224" \
        --log=error \
        --optypelist_for_implmode="Gelu" \
        --soc_version=Ascend${chip_name} \
        --op_select_implmode=high_performance \
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

    


## 推理验证

1. 对数据集推理  
    安装ais_bench推理工具。请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。完成安装后，执行以下命令预处理后的数据进行推理。
    ```bash
    python3 -m ais_bench \
        --model mae_batch_size${batch_size}.om \
        --input ./prep_dataset_batch_size1/ \ 
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
    python3 -m ais_bench --model mae_batch_size${batch_size}.om --batchsize ${batch_size}
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python3 MAE_postprocess.py \
        --folder-davinci-target ./result/ \
        --annotation-file-path ./ILSVRC2012/val_label.txt \
        --result-json-path ./result \
        --json-file-name result_batch_size1.json \
        --batch-size 1
    ```
    参数说明：
    + --folder-davinci-target: 存放推理结果的目录路径
    + --annotation-file-path: 标签文件路径
    + --result-json-path: 精度文件保存路径。
    + --json-file-name: 精度文件名。
    + --batch-size: 输入文件数量，当使用ais_bench工具推理时，参数为1。
    
    运行成功后，程序会将各top1~top5的正确率记录在 result_batch_size1.json 文件中，可执行以下命令查看：
    ```
    python3 -m json.tool result_batch_size1.json
    ```


----
# 性能&精度

在310P设备上，OM模型的精度为  **{Top1@Acc=83.52%}**，当batchsize设为1时模型性能最优，达 266.8 fps。

| 芯片型号   | BatchSize | 数据集      | 精度            | 性能       |
| --------- | --------- | ----------- | --------------- | --------- |
|Ascend310P3| 1         | ILSVRC2012  | Top1Acc=83.52% | 266.8 fps |
|Ascend310P3| 4         | ILSVRC2012  | Top1Acc=83.52% | 75.0 fps |
|Ascend310P3| 8         | ILSVRC2012  | Top1Acc=83.52% | 50.9 fps |
|Ascend310P3| 16        | ILSVRC2012  | Top1Acc=83.52% | 20.4 fps |
|Ascend310P3| 32        | ILSVRC2012  | Top1Acc=83.52% | 8.5 fps |
|Ascend310P3| 64        | ILSVRC2012  | Top1Acc=83.52% | 4.5 fps |