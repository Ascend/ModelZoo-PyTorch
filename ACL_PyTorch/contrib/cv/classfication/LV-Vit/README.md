# LV-Vit 模型推理指导

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

在本文中，提出了一个新的Vision Transformer训练方式称为LV-ViT，同时利用了patch token 和class token。该方法采用机器注释器生成的K维分数图作为监督，以密集方式监督所有token，其中K是目标数据集的类别数。通过这种方式，每个patch token显式地与指示相应图像patch内存在目标物体的单个位置特定监督相关联，从而在计算开销可以忽略不计的情况下提高vision Transformer的物体识别能力。这是首次证明密集监控有利于图像分类中的vision Transformer的工作。

+ 论文  
    [LV-Vit论文](https://arxiv.org/abs/2104.10858)
    Zihang Jiang, Qibin Hou, Li Yuan, Daquan Zhou, Yujun Shi, Xiaojie Jin, Anran Wang, Jiashi Feng

+ 参考实现：
    ```  
    https://github.com/zihangJiang/TokenLabeling.git
    branch:master  
    commit_id:2a217161fd5656312c8fac447fffbb6b3c091af7
    ```
## 输入输出数据
+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | actual_input_1 | FLOAT32 | NCHW | bs x 3 x 224 x 224 | 

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output1      |  FLOAT32   | ND          | bs x 1000        |


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
    cd LV-Vit
    
    git clone https://github.com/zihangJiang/TokenLabeling.git
    cd TokenLabeling
    patch -p1 < ../LV-Vit.patch
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
    ```
    mkdir -p prep_dataset
    python3 LV_Vit_preprocess.py --src_path ${datasets_path}/ILSVRC2012/images --save_path ./prep_dataset
    ```
    其中"datasets_path"表示处理前原数据集的地址，"prep_dataset"表示生成数据集的文件夹名称。

    
    运行后，将会得到如下形式的文件夹：

    ```
    ├── prep_dataset
    │    ├──input_00000.bin
    │    ├──......     	 
    ```


## 模型转换


1. 下载pth权重文件  
[LV-Vit预训练pth权重文件](https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_s-26M-224-83.3.pth.tar)  

    然后执行执行以下命令生成 ONNX 模型：
    ```bash
    python3 LV_Vit_pth2onnx.py --model_path ./model/lvvit_s-26M-224-83.3.pth.tar --onnx_path ./model/model_best_bs1.onnx --batch_size 1
    ```
    参数说明：
    + --model_path : 权重文件路径
    + --onnx_path : 输出onnx的文件路径
    + --batch_size ：输出onnx的可输入数据量


2. 使用 onnxsim 工具优化onnx模型  

    ```bash
    python3 -m onnxsim --input-shape="1,3,224,224" ./model/model_best_bs1.onnx ./model/model_best_bs1_sim.onnx
    ```
    参数说明：
    + --input-shape: 输入模型的宽高


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
    bs=1  # 根据需要自行设置 

    
    # 执行 ATC 进行模型转换
    atc --model=./model/model_best_bs1_sim.onnx \
        --framework=5 \
        --output=./model/model_best_bs1_sim \
        --input_format=NCHW \
        --input_shape="image:1,3,224,224" \
        --log=error \
        --soc_version=Ascend${chip_name} \
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
    该离线模型使用ais_infer作为推理工具，请参考[**安装文档**](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer#%E4%B8%80%E9%94%AE%E5%AE%89%E8%A3%85)安装推理后端包aclruntime与推理前端包ais_bench。完成安装后，执行以下命令预处理后的数据进行推理。
    ```bash
    python3 -m ais_bench \
        --model ./model/model_best_bs1_sim.om \
        --input ./prep_dataset/ \ 
        --output ./ \
        --output_dirname ./result/ \
        --outfmt TXT \
        --batchsize 1
    ```
    参数说明：
    + --model: OM模型路径
    + --input: 存放预处理后数据的目录路径
    + --output: 用于存放推理结果的父目录路径
    + --output:_dirname 用于存放推理结果的子目录名，位于--output指定的目录下
    + --outfmt: 推理结果文件的保存格式
    + --batchsize: 模型每次输入bin文件的数量,本例中为1。


2. 性能验证  
    对于性能的测试，需要注意以下三点：
    + 测试前，请通过`npu-smi info`命令查看NPU设备状态，请务必在NPU设备空闲的状态下进行性能测试。
    + 为了避免测试过程因持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps，反映模型在单位时间（1秒）内处理的样本数。
    ```bash
    python3 -m ais_bench --model ./model/model_best_bs1_sim.om --batchsize ${bs}
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python3 LV_Vit_postprocess.py --result_path ./result/dumpOutput_device0 --annotation_path ./ILSVRC2012/val.txt
    ```
    参数说明：
    + --result_path: 存放推理结果的目录路径
    + --annotation_path: 标签文件路径
    
    控制台输出如下信息

    ```
    accuracy: 0.8317
    ```
    


----
# 性能&精度

在310P设备上，OM模型的精度为  **{Top1Acc=83.17%}**，当batchsize设为4时模型性能最优，达 76.0 fps。

| 芯片型号   | BatchSize | 数据集      | 精度            | 性能       |
| --------- | --------- | ----------- | --------------- | --------- |
|Ascend310P3| 1         | ILSVRC2012  | Top1Acc=83.17%| 59.2 fps |
|Ascend310P3| 4         | ILSVRC2012  | Top1Acc=83.17%| 76.0 fps |
|Ascend310P3| 8         | ILSVRC2012  | Top1Acc=83.17%| 59.9 fps |
|Ascend310P3| 16        | ILSVRC2012  | Top1Acc=83.17%| 26.4 fps |
|Ascend310P3| 32        | ILSVRC2012  | Top1Acc=83.17%| 11.5 fps |
|Ascend310P3| 64        | ILSVRC2012  | Top1Acc=83.17%| 5.1 fps |