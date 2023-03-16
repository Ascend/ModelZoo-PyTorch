# SDMGR 模型推理指导

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

SDMGR是一种多模态端到端的文档关键信息抽取模型，它同时使用文本特征、文本框特征以及文本框间的空间位置关系解决关键信息提取问题。相比传统的做法，SDMGR具有更强大的鲁棒性和泛化性，可直接应用于一些全新文档版式的关键信息提取。此外，作者还构建了一个更具挑战性的文档关键信息提取数据集 WildReceipt。

+ 论文  
    [Spatial Dual-Modality Graph Reasoning for Key Information Extraction](https://arxiv.org/abs/2103.14470)  
    Hongbin Sun, Zhanghui Kuang, Xiaoyu Yue, Chenhao Lin, Wayne Zhang

+ 参考实现  
    ```
    url = https://github.com/open-mmlab/mmocr/blob/main/configs/kie/sdmgr/sdmgr_novisual_60e_wildreceipt.py
    tag = v0.6.3
    model_name = sdmgr_novisual_60e_wildreceipt
    ```
## 输入输出数据
该推理模型包含3个输入，2个输出。目前只支持每次输入一张图片的文本信息（relations, text, mask），经过推理后，输出文本的nodes与edges信息。该模型支持动态shape输入，具体的输入shape，只能在推理时根据实际输入数据的shape而确定。下面两张表中，num_text表示输入图片中包含多少条文本数据，而num_char表示每条文本包含字符数量的最大值。
+ 模型输入  
    | 输入名     | 数据类型   | 数据排布    | 大小                      |
    | ---------- | --------- | ---------- | ------------------------- |
    | relations  |  FLOAT32  | ND         | num_text x num_text x 5   |
    | texts      |  INT32    | ND         | num_text x num_char       |
    | mask       |  FLOAT32  | ND         | num_text x num_char x 256 |

+ 模型输出  
    | 输出名      | 数据类型    | 数据排布     | 大小                      |
    | ----------- | ---------- | ----------- | ------------------------- |
    | nodes       |  FLOAT32   | ND          | num_text x num_text       |
    | edges       |  FLOAT32   | ND          | (num_text * num_text) x 2 |



----
# 推理环境

- 该模型推理所需配套的软件如下：

    | 配套      | 版本    | 环境准备指导 |
    | --------- | ------- | ---------- |
    | 固件与驱动 | 1.0.18  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN      | 6.0.1   | -          |
    | Python    | 3.8.13  | -          |
    
    说明：请根据推理卡型号与 CANN 版本选择相匹配的固件与驱动版本。


----
# 快速上手

## 获取源码

1. 克隆开源仓源码并进行修改
    ```bash
    git clone -b v0.6.3 https://github.com/open-mmlab/mmocr.git
    patch -p1 < sdmgr.patch
    ```

2. 安装推理过程所需的依赖
    ```bash
    pip3 install torch==1.13.0 torchaudio torchvision
    pip3 install openmim
    mim install mmcv-full==1.7.0
    mim install mmdet==2.25.3
    pip3 install decorator 
    pip3 install sympy
    mim install -v -e ./mmocr/
    pip3 install numpy==1.23.5
    pip3 install tqdm
    ```
    说明：若安装过程中因网络原因报错，可在报错的命令尾部追加参数`-i`指定国内源，然后重新执行这条命令，例如：
    ```bash
    pip3 install torch==1.13.0 torchaudio torchvision -i https://repo.huaweicloud.com/repository/pypi/simple/
    mim install mmdet==2.25.3 -i https://repo.huaweicloud.com/repository/pypi/simple/
    mim install -v -e ./mmocr/ -i https://repo.huaweicloud.com/repository/pypi/simple/
    ```

3. 创建一个目录，用于存放整个推理过程中所需文件与生成文件
    ```bash
    mkdir sdmgr
    ```

## 准备数据集

1. 获取原始数据集  
    该模型使用 [WildReceipt](https://paperswithcode.com/dataset/wildreceipt) 测试集中的472条文本信息来验证精度。参考以下命令下载数据集并解压。
    ```bash
    wget https://download.openmmlab.com/mmocr/data/wildreceipt.tar -P data/
    tar -xvf data/wildreceipt.tar -C data/
    ```

    执行完上述步骤后，在当前目录下的数据目录结构为：   
    ```
    data/wildreceipt/
            ├── class_list.txt
            ├── dict.txt
            ├── image_files/
                └── ...
            ├── test.txt
            └── train.txt
    ```


2. 数据预处理  
    执行前处理脚本将原始数据转换为离线推理支持的npy文件。
    ```bash
    python3 sdmgr_preprocess.py \
        --config mmocr/configs/kie/sdmgr/sdmgr_novisual_60e_wildreceipt.py \
        --save-dir sdmgr/prep_npy/
    ```
    参数说明：
    + --config: 模型配置文件路径
    + --save-dir: 存放生成的bin/npy文件的目录路径
    
    运行成功后，sdmgr/prep_npy/ 目录下会生成3个子目录，分别对应模型的3个输入。每个子目录下都会生成472个npy文件。


## 模型转换

1. PyTroch 模型转 ONNX 模型  

    step1: 下载预训练权重文件
    下载mmocr提供的[预训练模型](https://download.openmmlab.com/mmocr/kie/sdmgr/sdmgr_novisual_60e_wildreceipt_20220803-d06d4a1a.pth) 到 sdmgr 目录下，可参考命令：
    ```bash
    wget https://download.openmmlab.com/mmocr/kie/sdmgr/sdmgr_novisual_60e_wildreceipt_20220803-d06d4a1a.pth -P sdmgr
    ```

    step2: 生成 ONNX 模型
    ```bash
    python3 sdmgr_pth2onnx.py \
        --config mmocr/configs/kie/sdmgr/sdmgr_novisual_60e_wildreceipt.py \
        --checkpoint sdmgr/sdmgr_novisual_60e_wildreceipt_20220803-d06d4a1a.pth \
        --prep-dir sdmgr/prep_npy/ \
        --onnx sdmgr/sdmgr_novisual_wildreceipt.onnx \
        --opset-version 12
    ```
    参数说明：
    + --config: 模型配置文件路径
    + --checkpoint: 预训练权重文件的路径
    + --onnx: 生成ONNX模型的保存路径
    + --opset-version: ONNX算子集版本，默认为12
    
    注：opset-version建议设为12，其他版本会导致模型转换失败或无法推理，后期修复成本较大。

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
    然后根据回显结果设置chip_name:  
    ```bash
    chip_name=310P3
    ```

    step2: ONNX 模型转 OM 模型
    ```bash
    # 配置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /etc/profile
    
    # 执行 ATC 进行模型转换
    atc --framework=5 \
        --model=sdmgr/sdmgr_novisual_wildreceipt.onnx \
        --input_shape_range="relations:[1~196,1~196,5];texts:[1~196,1~59];mask:[1~196,1~59,256]" \
        --input_format=ND \
        --output=sdmgr/sdmgr_novisual_wildreceipt \
        --log=error \
        --soc_version=Ascend${chip_name} \
        --keep_dtype=keep_dtype.cfg
    ```
    
    参数说明：
    + --framework: 5代表ONNX模型
    + --model: ONNX模型路径
    + --input_shape_range: 模型各输入数据的shape
    + --input_format: 输入数据的排布格式
    + --output: OM模型路径，无需加后缀
    + --log：日志级别
    + --soc_version: 处理器型号
    + --keep_dtype：指定转换过程中需要保持原精度计算的算子名
    
    注：ONNX模型中的`/gnn_layers.0/Mul_2`节点，其输入包含一个数值为1000000000的Initializer，在fp32精度范围内可正常表示。若按照常规流程转OM后，全网浮点数精度变为fp16，而在fp16下表示数值1000000000会发生溢出，进而导致OM模型精度异常。因此，这里使用了`--keep_dtype`参数来避免此问题。

## 推理验证

1. 离线推理&性能验证

    该离线模型使用ais_bench作为推理工具，推理前，请参考[**安装文档**](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)安装推理后端包aclruntime与推理前端包ais_bench。  
    该模型各输入的首轴均为动态，若使用原生推理命令，程序无法正确获取batch_size，导致统计出的性能数据不可信。因此，这里调用[ais_bench的推理接口](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench#%E6%8E%A5%E5%8F%A3%E5%BC%80%E6%94%BE)，使用python的time库统计推理耗时，从而推算吞吐率。

    运行推理脚本对预处理后的数据进行推理：
    ```bash
    python3 sdmgr_inference.py \
        --model sdmgr/sdmgr_novisual_wildreceipt.om \
        --input sdmgr/prep_npy/relations,sdmgr/prep_npy/texts,sdmgr/prep_npy/mask \
        --output sdmgr/inference_output/
    ```
    参数说明：
    + --model 模型路径
    + --input 存放预处理后数据的目录路径
    + --output 用于存放推理结果的目录路径
    
    运行结束后，`sdmgr/inference_output/`目录下会生成推理结果。同时，会打印出模型的性能数据：
    ```
    [INFO] ----------------------Performance Summary-----------------------
    [INFO] Total time: 5763.430 ms.
    [INFO] Average time without first time: 12.186 ms.
    [INFO] Throughput: 82.063 fps.
    [INFO] ----------------------------------------------------------------
    ```

2. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python3 sdmgr_postprocess.py \
        --config mmocr/configs/kie/sdmgr/sdmgr_novisual_60e_wildreceipt.py \
        --res-dir sdmgr/inference_output/
    ```
    参数说明：
    + --config: 模型配置文件路径
    + --res-dir: 存放推理结果的目录路径
    
    运行成功后，程序会打印出模型的精度指标：
    ```
    {'macro_f1': 0.87049395}
    ```


----
# 性能&精度

在310P设备上，OM模型的精度与目标精度[ Macro_F1 = 0.871 ](https://github.com/open-mmlab/mmocr/tree/main/configs/kie/sdmgr#wildreceipt)的相对误差低于 1%，性能达82.063fps。

| 芯片型号   | 数据集      | 精度            | 性能       |
| --------- | ----------- | --------------- | --------- |
|Ascend310P3| WildReceipt | Macro_F1=0.870 | 82.063fps |

