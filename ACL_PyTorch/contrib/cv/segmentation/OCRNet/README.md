# OCRNet 模型推理指导

- [概述](#概述)
    - [输入输出数据](#输入输出数据)
- [推理环境](#推理环境)
- [快速上手](#快速上手)
    - [获取源码](#获取源码)
    - [准备数据集](#准备数据集)
    - [模型转换](#模型转换)
    - [推理验证](#推理验证)
- [性能&精度](#模型推理性能&精度)

----
# 概述

通过研究语义分割中的上下文聚合问题。基于像素的标签是像素所属对象的类别，提出了一种简单而有效的方法OCRNet，即对象上下文表示，通过利用相应对象类的表示来表征像素。首先，在地面真值分割的监督下学习目标区域。其次，通过聚集对象区域中像素的表示来计算对象区域的表示。最后，计算每个像素和每个目标区域之间的关系，并用对象上下文表示来增强每个像素的表示，这是所有对象区域表示的加权聚合。实验表明，提出的方法在不同的基准点上取得了具有竞争力的表现。
+ 论文  
​[Segmentation Transformer: Object-Contextual Representations for Semantic Segmentation](https://arxiv.org/abs/1909.11065v6): Yuhui Yuan  , Xiaokang Chen  , Xilin Chen  , and Jingdong Wang 

+ 参考实现  
    ```
    url=https://github.com/PaddlePaddle/PaddleSeg.git
    branch=remotes/origin/release/2.1
    ```
## 输入输出数据
+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | input1  |  FLOAT32  | NCHW         | batchsize x 3 x 1024 x 1024   |

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output1       |  FLOAT32   | NCHW          | batch_size x 1024 x 2048       |


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

1. 安装依赖。。
    ```bash
    pip3 install -r requirements.txt
    ```
    

2. 获取源码并安装。
    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    git clone https://github.com/PaddlePaddle/PaddleSeg.git
    cd PaddleSeg
    git checkout remotes/origin/release/2.1
    ```


## 准备数据集

1. 获取原始数据集  
    用户需自行获取CityScapes数据集中的验证集部分（无需训练集），上传数据集到服务器中。其内容如下所示：    
    ```
    cityscapes
        ├─gtFine
        │  └─val
        │      ├─frankfurt
        │      │      frankfurt_000000_000294_gtFine_color.png
        │      │      frankfurt_000000_000294_gtFine_instanceIds.png
        │      │      frankfurt_000000_000294_gtFine_labelIds.png
        │      │      frankfurt_000000_000294_gtFine_labelTrainIds.png
        │      │      frankfurt_000000_000294_gtFine_polygons.json
        │      ├─lindau
        │      └─munster
        └─leftImg8bit
            └─val
                ├─frankfurt
                │      frankfurt_000000_000294_leftImg8bit.png
                ├─lindau
                │      lindau_000000_000019_leftImg8bit.png
                └─munster
                        munster_000000_000019_leftImg8bit.png

    ```


2. 数据预处理  
    数据预处理，将原始数据集转换为模型输入的数据。
    执行OCRNet_preprocess.py脚本，完成预处理。
    ```bash
    python3 OCRNet_preprocess.py --src_path /opt/npu/cityscapes/ --bin_file_path bs1_bin --batch_size 1
    ```
    参数说明：
    + --src_path: 原始数据验证集所在路径。
    + --bin_file_path: 预处理结果输出目录。
    + --batch_size: 输入的数据量。
    
    运行成功后，生成“bs1_bin”文件夹，bs1_bin目录下生成的是供模型推理的bin文件。


## 模型转换

1. PyTroch 模型转 ONNX 模型  
    step1: 获取权重文件[OCRNet权重文件下载链接](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ocrnet_hrnetw18_cityscapes_1024x512_160k/model.pdparams)放在根目录下
    step2: 导出onnx文件。

    ```bash
    config_path=PaddleSeg/configs/ocrnet/ocrnet_hrnetw18_cityscapes_1024x512_160k.yml
    pd_model=pd_model
    pdparams_path=model.pdparams

    python3 PaddleSeg/export.py \
        --save_dir ${pd_model} \
        --model_path ${pdparams_path} \
        --config ${config_path} 
    ```
    参数说明：
    + --save_dir: 保存pb文件路径
    + --model_path：pdparams权重文件路径
    + --config：模型配置文件

    ```bash
    paddle2onnx \
        --model_dir ${pd_model} \
        --model_filename ${pd_model}/model.pdmodel \
        --params_filename ${pd_model}/model.pdiparams \
        --save_file ${onnx_path} \
        --opset_version 11
    ```
    参数说明：
    + --model_dir: 保存pb文件路径
    + --model_filename：pb_model文件名称。
    + --params_filename：pdiparams文件名称
    + --save_file：onnx文件名称
    + --opset_version：opset版本

    step3:优化onnx文件
    ```
    batch_size=1
    python3 -m onnxsim ocrnet.onnx onnx/ocrnet_bs${batch_size}.onnx \
        --input-shape="x:${batch_size},3,1024,2048" \
        --skip-fuse-bn 
    python3 optimize_onnx.py onnx/ocrnet_bs${batch_size}.onnx \
        onnx/ocrnet_optimize_bs${batch_size}.onnx
    ```
2. ONNX 模型转 OM 模型  
    此步骤只能在NPU设备上进行，所以执行atc命令转换模型前，需将上一步生成的ONNX复制到NPU设备。

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
    source /etc/profile
    
    batch_size=1
    chip_name=310P3
    
    # 执行 ATC 进行模型转换
    atc --model=./ocrnet_rjy_fix.onnx \
        --framework=5 \
        --output=./ocrnet_optimize_bs${batch_size} \
        --input_format=NCHW \
        --input_shape="x:1,3,1024,2048" \
        --soc_version=Ascend310${chip_name} 
    ```

   参数说明：
    + --framework: 5代表ONNX模型
    + --model: ONNX模型路径
    + --input_shape: 模型输入数据的shape
    + --input_format: 输入数据的排布格式
    + --output: OM模型路径，无需加后缀
    + --soc_version: 处理器型号


## 推理验证

1. 该离线模型使用ais_infer作为推理工具，请参考[**安装文档**](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench#%E4%B8%80%E9%94%AE%E5%AE%89%E8%A3%85)安装推理后端包aclruntime与推理前端包ais_bench。完成安装后，执行以下命令预处理后的数据进行推理。
    ```bash
    python3 -m ais_bench
        --model ./ocrnet_optimize_bs${batch_size}.om  \
        --input ./bs${batch_size}_bin/imgs \
        --output result  \
        --output_dirname ./outputs_bs${batch_size}_om 
        --batchsize ${batch_size}
    ```
    参数说明：
    + --model: OM模型路径
    + --input: 存放预处理后数据的目录路径
    + --output: 用于存放推理结果的父目录路径
    + --output_dirname: 用于存放推理结果的子目录路径，位于--output指定的目录下
    + --batchsize: 模型一次处理多少样本

2. 性能验证  

    对于性能的测试，需要注意以下三点：
    + 测试前，请通过`npu-smi info`命令查看NPU设备状态，请务必在NPU设备空闲的状态下进行性能测试。
    + 为了避免测试过程因持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps，反映模型在单位时间（1秒）内处理的样本数。
    ```bash
    python3 -m ais_bench --model ./ocrnet_optimize_bs${batch_size}.om --loop 100 --batchsize ${batch_size}
    ```
    
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  
    
    此步骤需要将NPU服务器上OM模型的推理结果复制到GPU服务器上，然后再GPU服务器上执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python3 OCRNet_postprocess.py \
        --bin_file_path bs1_bin \
        --pred_path $path
    ```
    参数说明：
    + --bin_file_path: 预处理数据集路径
    + --pred_path: 推理结果文件路径
    

## 模型推理性能&精度

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集       | 精度        | 性能               |
| -------- | ---------- | ------------ | ----------- | ------------------ |
| 310P3    | 1          | Imagenet2012 | miou:79.63% | 13.3 |
| 310P3    | 4          | Imagenet2012 | miou:79.63% | 10.3 |
| 310P3    | 8          | Imagenet2012 | miou:79.63% | 9.0 |
| 310P3    | 16         | Imagenet2012 | miou:79.63% | 9.3 |


