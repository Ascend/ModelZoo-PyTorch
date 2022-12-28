# Convmixer_1536_20 模型推理指导

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

ConMixer在思想上类似于ViT和MLP-Mixer，它直接将patch作为输入，分离空间和通道尺寸的混合建模，并在整个网络中保持相同大小的分辨率。但是ConvMixer只使用标准卷积来实现混合步骤。尽管ConvMixer的设计很简单，但是实验证明了ConvMixer在相似的参数计数和数据集大小方面优于ViT、MLP-Mixer及其一些变体，以及经典的视觉模型，如ResNet。
+ 论文  
    [​Patches Are All You Need](https://openreview.net/forum?id=TVHS5Y4dNvM): Asher Trockman, J Zico Kolter.(2021)

+ 参考实现  
    ```
    url= https://github.com/locuslab/convmixer.git
    branch=master
    commit_id=d5fb538e06251ece53fdd3f7a37ebaec0c0ae4ee
    model_name=convmixer_1536_20
    ```
## 输入输出数据
+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | input1  |  FLOAT32  | NCHW         | batchsize x 3 x 224 x 224   |

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output1       |  FLOAT32   | ND          | batch_size x 1000       |


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
    git clone https://github.com/locuslab/convmixer.git
    mv convmixer_patch.patch convmixer
    cd convmixer
    git apply convmixer_patch.patch
    scp convmixer.py ../convmixer_net.py
    cd ..
    ```


## 准备数据集

1. 获取原始数据集  
    ​获取Imagenet数据集：imagenet2012，下载其中ILSVRC2012/图片及其标注文件（images， val_label.txt），将数据集置于convmixer_1536_20根目录下，sdsad数据集目录结构如下：
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
    数据预处理，将原始数据集转换为模型输入的数据。
    执行convmixer_preprocess.py脚本，完成预处理。
    ```bash
    python3 convmixer_preprocess.py \
    ​    --image-path ${datasets_path}/images/ \
    ​    --prep-image ./prep_image_bs${batch_size} \
    ​    --batch-size ${batch_size}
    ```
    参数说明：
    + --datasets_path: 原始数据验证集所在路径。
    + --batch_size: 每个后处理文件所包含的数据量。
    
    
    运行成功后，生成“prep_image_bs”文件夹，prep_image_bs目录下生成的是供模型推理的bin文件。


## 模型转换

1. PyTroch 模型转 ONNX 模型  
    step1: 获取权重文件[convmixer_1536_20_ks9_p7.pth.tar](https://github.com/tmp-iclr/convmixer/releases/download/v1.0/convmixer_1536_20_ks9_p7.pth.tar)放在根目录下
    step2: 导出onnx文件。

    ```bash
    python3 convmixer_pth2onnx.py \
        --source "./convmixer_1536_20_ks9_p7.pth.tar" \
        --target "./convmixer_1536_20.onnx"
    ```
    参数说明：
    + --source : 预训练权重文件的路径
    + --target: 生成ONNX模型的保存路径

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
    atc --model=./convmixer_1536_20.onnx \
        --framework=5 \
        --output=./convmixer_1536_20_${batch_size} \
        --input_format=NCHW \
        --input_shape="image:1,3,224,224" \
        --soc_version=Ascend310${chip_name} \
        --op_select_implmode=high_performance \
        --optypelist_for_implmode="Gelu"
    ```

   参数说明：
    + --framework: 5代表ONNX模型
    + --model: ONNX模型路径
    + --input_shape: 模型输入数据的shape
    + --input_format: 输入数据的排布格式
    + --output: OM模型路径，无需加后缀
    + --soc_version: 处理器型号


## 推理验证

1. 安装ais_bench推理工具，请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。完成安装后，执行以下命令预处理后的数据进行推理。
    ```bash
    python3 -m ais_bench
        --model ./convmixer_1536_20_${batch_size} \
        --input ./data/kinetics-skeleton/ \
        --output result  \
        --output_dirname ./st_gcn_bs${bs}_out
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
    python3 -m ais_bench --model ./convmixer_1536_20_${batch_size}.om --loop 100 --batchsize ${batch_size}
    ```
    
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  
    
    此步骤需要将NPU服务器上OM模型的推理结果复制到GPU服务器上，然后再GPU服务器上执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python3 convmixer_eval_acc.py --folder-davinci-target ./result/outputs_bs1_om/ --annotation-file-path ./ILSVRC2012/val_label.txt --result-json-path ./result --json-file-name result_bs1.json --batch-size ${batch_size}
    ```
    参数说明：
    + --result_dir: 存放推理结果的目录路径
    + --label_path: 标签文件所在路径
    
    运行成功后，程序会打印出模型的精度指标：
    ```
    top1:81.37%
    ```


## 模型推理性能&精度

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集       | 精度        | 性能               |
| -------- | ---------- | ------------ | ----------- | ------------------ |
| 310P3    | 1          | Imagenet2012 | top1:81.37% | 102.91136243948735 |
| 310P3    | 4          | Imagenet2012 | top1:81.37% | 95.795763362348555 |

说明：

Top1表示预测结果中概率最大的类别与真实类别一致的概率，其值越大说明分类模型的效果越优

