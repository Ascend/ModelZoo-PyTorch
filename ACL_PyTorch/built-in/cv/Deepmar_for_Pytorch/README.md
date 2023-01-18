#   Deepmar 模型推理指导

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

就行人属性识别领域存在的两个主要问题（手工找特征不能很好的适用视频场景、属性之间的关系被忽略），主要提出了两个网络，DeepSAR和DeepMAR。
DeepSAR：独立识别每个属性。将每一个属性的识别当作二元分类问题，然后一个一个识别每个属性。DeepMAR：利用属性之间的关系，如长发更有可能是女性，所以头发的长度有利于帮助识别性别属性。将所有属性的识别一次性完成，多标签分类问题。

+ 论文  
    [deepmar论文](https://ieeexplore.ieee.org/document/7486476)
    Dangwei Li; Xiaotang Chen; Kaiqi Huang

+ 参考实现：
    ```  
    https://github.com/dangweili/pedestrian-attribute-recognition-pytorch.git
    branch:master  
    commit_id:468ae58cf49d09931788f378e4b3d4cc2f171c22
    ```
## 输入输出数据
+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | actual_input_1 | FLOAT32 | NCHW | bs x 3 x 224 x 224 | 

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output1      |  FLOAT32   | ND          | bs x 35        |

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
    git clone https://github.com/dangweili/pedestrian-attribute-recognition-pytorch
    cd pedestrian-attribute-recognition-pytorch
    git reset --hard 468ae58cf49d09931788f378e4b3d4cc2f171c22
    patch -p1 < ../deepmar.patch
    ```

## 准备数据集

1. 获取原始数据集  
    该模型使用PETA数据集，请用户需自行获取PETA数据集的19000张图片，并取出其中7600张图片作为测试集。可以从Deepmar开源仓中下载数据集。



2. 数据预处理  

    数据预处理将原始数据集转换为模型输入二进制格式。通过缩放、均值方差手段归一化，输出为二进制文件。
    执行preprocess_deepmar_pytorch.py脚本，保存图像数据到bin文件。
    ```
    python3 preprocess_deepmar_pytorch.py --file_path /home/HwHiAiUser/dataset/peta/images --bin_path input_bin --image_info image.txt
    ```
    参数说明：
    + --file_path : 原始数据路路径
    + --bin_path : 输出二进制文件路径
    + --image_info ：图片信息文件路径
    执行成功后应在指定目录下产生含有bin文件的文件夹


## 模型转换


1. 下载pth权重文件  
[Deepmar预训练pth权重文件](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/turing/resourcecenter/model/ATC%20Deepmar%20from%20Pytorch%20Ascend310/zh/1.1/ATC_Deepmar_from_Pytorch_Ascend310.zip)  
解压文件夹后，获得checkpoint.pth.tar
    
    1.将提供的export_onnx.py放入开源仓“pedestrian-attribute-recognition-pytorch”目录下。
    
    2.在“pedestrian-attribute-recognition-pytorch”目录下，执行export_onnx.py脚本将.pth.tar文件转换为.onnx文件，执行如下命令。
    ```
    python3 export_onnx.py --input_file xxx/checkpoint.pth.tar --output_file deepmar_bs1.onnx --batch_size 1
    ```
    参数说明：
    + --input_file : 权重文件路径
    + --output_file : 输出onnx的路径
    + --batch_size ：模型输入量

    运行成功后，在当前目录生成deepmar_bs1.onnx模型文件，然后将deepmar_bs1.onnx	复制到deepmar源码包中。
     **说明：**  
    >注意目前ATC支持的onnx算子版本为11


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
    step2: 本模型中的pad算子没有用，可以使用remove_pad.py脚本剔除，提升部分性能。
    ```bash
    # 配置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 remove_pad.py --model_name deepmar_bs1.onnx --output_name deepmar_bs1_nopad.onnx
    ```
   参数说明：
    + --model_name : 输入onnx文件路径
    + --output_name : 出书onnx的文件路径

    step3: ONNX 模型转 OM 模型
    ```bash
    # 配置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    chip_name=310P3  # 根据 step1 的结果设值 

    # 执行 ATC 进行模型转换
    atc --model=./deepmar_bs1_nopad.onnx \
        --framework=5 \
        --output=./deepmar_bs1 \
        --input_format=NCHW \
        --input_shape="actual_input_1:1,3,224,224" \
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
    该离线模型使用ais_infer作为推理工具，请参考[**安装文档**](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer#%E4%B8%80%E9%94%AE%E5%AE%89%E8%A3%85)安装推理后端包aclruntime与推理前端包ais_bench。完成安装后，执行以下命令预处理后的数据进行推理。
    ```bash
    python3 -m ais_bench \
        --model ./deepmar_bs1.om \
        --input ./input_bin/ \ 
        --output ./result/ \
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
    python3 -m ais_bench --model ./deepmar_bs1.om --batchsize ${bs}
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的精度：
    调用postprocess_deepmar_pytorch.py脚本与数据集标签label.json比对，可以获得Accuracy数据，结果保存在fusion_result.json中。
    ```
    python3 postprocess_deepmar_pytorch.py --npu_result result/dumpOutput_device0/ --label_file label.json
    ```
    参数说明：
    + --npu_result : 推理结果路径
    + --label_file : 标签文件路径

    控制台输出如下信息

    ```
    instance_acc: 0.78965245043699
    instance_precision: 0.8823281028182345
    instance_recall: 0.8496866930584036
    instance_F1: 0.8656998192635741
    ```
    


----
# 性能&精度

在310P设备上，OM模型的精度为  **{Acc=78.9%}**，当batchsize设为1时模型性能最优，达 1642.5 fps。

| 芯片型号   | BatchSize | 数据集      | 精度            | 性能       |
| --------- | --------- | ----------- | --------------- | --------- |
|Ascend310P3| 1         | PETA  | Acc=78.9%| 1642.5 fps |
|Ascend310P3| 4         | PETA  | Acc=78.9%| 869.4 fps |
|Ascend310P3| 8         | PETA  | Acc=78.9%| 497.7 fps |
|Ascend310P3| 16        | PETA  | Acc=78.9%| 252.4 fps |
|Ascend310P3| 32        | PETA  | Acc=78.9%| 80.6 fps |
|Ascend310P3| 64        | PETA  | Acc=78.9%| 47.5 fps |