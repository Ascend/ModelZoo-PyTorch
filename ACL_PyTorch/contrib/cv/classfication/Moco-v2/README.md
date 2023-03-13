# Moco-v2 模型推理指导

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

通过在MoCo框架中实现SimCLR的两个设计改进来验证其有效性，通过对MoCo的简单修改——即使用MLP投影头和更多的数据增强——建立了比SimCLR性能更好的更强的基线，并且不需要大规模的批量训练，SimCLR中使用的两个设计改进，即 MLP投影头 和 更强的数据增强，与MoCo和SimCLR框架正交，当与MoCo一起使用时，它们会带来更好的图像分类和目标检测迁移学习结果

+ 论文  
    [Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/abs/2003.04297)  
    Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He

+ 参考实现： 
    ``` 
    https://github.com/facebookresearch/moco
    branch:master  
    commit_id:78b69cafae80bc74cd1a89ac3fb365dc20d157d3
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
    | torch     | 1.5.0   | -          |    
    说明：请根据推理卡型号与 CANN 版本选择相匹配的固件与驱动版本。


----
# 快速上手

## 安装

- 安装推理过程所需的依赖
    ```bash
    pip3 install -r requirements.txt
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
    python3 imagenet_torch_preprocess.py --input_img_dir ${datasets_path}/ILSVRC2012/images --output_img_dir ./prep_dataset
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
[moco-v2预训练pth权重文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/MocoV2/PTH/model_lincls_best.pth.tar)  

    然后执行执行以下命令生成 ONNX 模型：
    ```bash
    python3 pthtar2onnx.py --bs 1 --weight model_lincls_best.pth.tar
    ```
    参数说明：
    + --bs: 输入模型的数据量
    + --weight: 输入的pth模型

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
    atc --model=./moco-v2-bs1.onnx \
        --framework=5 \
        --output=moco-v2-atc-${bs} \
        --input_format=NCHW \
        --input_shape="actual_input_1:1,3,224,224" \
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
   请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。
    ```bash
    python3 -m ais_bench \
        --model moco-v2-atc-${bs} \
        --input ./ prep_dataset/ \ 
        --output ./ \
        --output_dirname ./result/ \
        --outfmt TXT \
        --batchsize ${bs}
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
    python3 -m ais_bench --model moco-v2-atc-bs1 --batchsize ${bs}
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python3 vision_metric_ImageNet.py  --folder-davinci-target result/dumpOutput_device0/ --annotation-file-path ./val_label.txt --result-json-path ./ --json-file-name result.json
    ```
    参数说明：
    + --folder-davinci-target: 存放推理结果的目录路径
    + --annotation-file-path: 标签文件路径
    + --result-json-path: 精度文件保存路径。
    + --json-file-name: 精度文件名。
    
    运行成功后，程序会将各top1~top5的正确率记录在 result_bs1.json 文件中，可执行以下命令查看：
    ```
    python3 -m json.tool result.json
    ```


----
# 性能&精度

在310P设备上，OM模型的精度为  **{Top1Acc=67.41% Top5Acc=88.02%}**，当batchsize设为4时模型性能最优，达 836.0 fps。

| 芯片型号   | BatchSize | 数据集      | 精度            | 性能       |
| --------- | --------- | ----------- | --------------- | --------- |
|Ascend310P3| 1         | ILSVRC2012  | Top1Acc=67.41% Top5Acc=88.02%| 836.2 fps |
|Ascend310P3| 4         | ILSVRC2012  | Top1Acc=67.41% Top5Acc=88.02%| 836.0 fps |
|Ascend310P3| 8         | ILSVRC2012  | Top1Acc=67.41% Top5Acc=88.02%| 497.9 fps |
|Ascend310P3| 16        | ILSVRC2012  | Top1Acc=67.41% Top5Acc=88.02%| 244.4 fps |
|Ascend310P3| 32        | ILSVRC2012  | Top1Acc=67.41% Top5Acc=88.02%| 79.5 fps |
|Ascend310P3| 64        | ILSVRC2012  | Top1Acc=67.41% Top5Acc=88.02%| 39.5 fps |