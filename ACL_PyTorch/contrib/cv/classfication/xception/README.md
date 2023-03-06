#  xception 模型推理指导

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

Xception是Google公司继Inception后提出的对 Inception-v3 的另一种改进。作者认为，通道之间的相关性与空间相关性最好要分开处理。于是采用 Separable Convolution来替换原来 Inception-v3中的卷积操作。

+ 论文  
    [xception论文](https://arxiv.org/abs/1610.02357)  
    François Chollet

+ 参考实现：  
    [xception代码](https://github.com/tstandley/Xception-PyTorch)  

## 输入输出数据
+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | image | FLOAT32 | NCHW | BATCHS_SIZE,3,299,299 | 

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output1      |  FLOAT32   | NCHW          | BATCHS_SIZE,1000        |


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
    git clone https://github.com/tstandley/Xception-PyTorch
    cd Xception-PyTorch  
    git reset 7b9718bb525fefc95f507306e685aa8998d0492c --hard  
    cd ..
    ```
如果使用补丁文件修改了模型代码则将补丁打入模型代码，如果需要引用模型代码仓的类或函数通过sys.path.append(r"./Xception-PyTorch")添加搜索路径。

## 准备数据集

1. 获取原始数据集  
    该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/root/dataset/ILSVRC2012//val与/root/dataset/ILSVRC2012/val_label.txt。 



2. 数据预处理  
    执行前处理脚本将原始数据转换为OM模型输入需要的bin/npy文件。
    ```bash
    python3 img_preprocess.py --src_path /root/dataset/ILSVRC2012/images/ --save_path ./pre_dataset
    ```
    其中"src_path"表示处理前原数据集的地址，"save_path"表示生成数据集的文件夹名称

    运行后，将会得到如下形式的文件夹：

    ```
    ├── pre_dataset
    │    ├──ILSVRC2012_val_00000003.bin
    │    ├──......     	 
    ```


## 模型转换

1. PyTroch 模型转 ONNX 模型  

    xception预训练pth[权重文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Xception/PTH/xception-c0a72b38.pth.tar)
 

    然后执行执行以下命令生成 ONNX 模型：
    ```
    python3 xception_pth2onnx.py  --input_file xception-c0a72b38.pth.tar  --output_file xception.onnx
    ```
    参数说明：
     + --input_file: 参数配置文件路径
     + --output_file: 生成ONNX模型的保存路径


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
    # 执行 ATC 进行模型转换
    atc --framework=5 --model=xception.onnx --output=xception_1 --input_format=NCHW --input_shape="image:1,3,299,299" --log=debug  --soc_version=Ascend${chip_name}
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
    mkdir results
    python3 -m ais_bench \
        --model ./xception_1.om \
        --input ./pre_dataset \
        --output ./results \
        --outfmt BIN \
        --batchsize 1
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
    python3 -m ais_bench --model xception_1.om --batchsize 1
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python3 imagenet_acc_eval.py --folder_davinci_target ./results/****** --annotation_file_path /root/dataset/ILSVRC2012/val_label.txt --result_json_path ./ --json_file_name result.json
    ```
    参数说明：
    + --folder_davinci_target: 存放推理结果的目录路径
    + --annotation_file_path: 标签文件路径
    + --result_json_path: 后处理结果路径。
    + --json_file_name: 推理保存节点数。
    运行成功后，程序会将各top1~top5的正确率记录在 result.json 文件中，可执行以下命令查看：
    ```

----
# 性能&精度

在310P设备上，OM模型的精度为  **{top1:78.8%;top5:94.33%}**，当batchsize设为1时模型性能最优，达 22.06 fps。

| 芯片型号   | BatchSize | 数据集      | 精度            | 性能       |
| --------- | --------- | ----------- | --------------- | --------- |
|Ascend310P3| 1         | imagenet  | top1:78.8%;top5:94.33% | 799.4563 fps |
|Ascend310P3| 4         | imagenet  | top1:78.8%;top5:94.33% | 1374.5468 fps |
|Ascend310P3| 8         | imagenet  | top1:78.8%;top5:94.33% | 1491.660 fps |
|Ascend310P3| 16         | imagenet  | top1:78.8%;top5:94.33% | 1506.938 fps |
|Ascend310P3| 32        | imagenet  | top1:78.8%;top5:94.33% | 1424.2953 fps |
|Ascend310P3| 64         | imagenet  | top1:78.8%;top5:94.33% | 829.0842 fps |




