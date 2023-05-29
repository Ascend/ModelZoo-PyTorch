# BiLSTM_CRF(PyTorch) 模型推理指导

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

CLUENER 细粒度命名实体识别

+ 论文  
    [CLUENER2020: Fine-grained Named Entity Recognition Dataset and Benchmark for Chinese](https://arxiv.org/abs/2001.04351)  
    Liang Xu, Yu tong, Qianqian Dong, Yixuan Liao, Cong Yu, Yin Tian, Weitang Liu, Lu Li, Caiquan Liu, Xuanwei Zhang 

+ 参考实现：  
    url = https://github.com/CLUEbenchmark/CLUENER2020/tree/master/bilstm_crf_pytorch  
    branch = master  
    commit_id = 2e8fccd1b8cde6b471fe440b9d766e920fb418ab  

## 输入输出数据
+ 模型输入  
    | 输入节点名 | 数据类型 | 数据排布 | 数据尺寸 |
    | --------- | -------- | ------- | ------- |
    | ids       | INT64    | ND      | bs x 50 | 
    | mask      | INT64    | ND      | bs x 50 | 

+ 模型输出  
    | 输出节点名 | 数据类型 | 数据排布 | 数据尺寸      |
    | --------- | ---------- | ----- | ------------ |
    | features  |  FLOAT32   | ND    | bs x 50 x 33 |


----
# 推理环境

- 该模型推理所需配套的软件如下：

    | 配套      | 版本    | 环境准备指导 |
    | --------- | ------- | ---------- |
    | 固件与驱动 | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN      | 6.0.1   | -          |
    | Python    | 3.8.16  | -          |
    
    说明：请根据推理卡型号与 CANN 版本选择相匹配的固件与驱动版本。


----
# 快速上手

  + 全局设置：  
    ```bash
    mkdir -p ./inference   # 创建一个目录，用于存放推理过程中生成的各种文件
    max_seq_len=50    # 设置模型可处理的最大文本长度
    ```    

## 获取源码

1. 安装推理过程所需的依赖
    ```bash
    pip install -r requirements.txt
    ```
2. 获取开源仓源码
    ```bash
    git clone https://github.com/CLUEbenchmark/CLUENER2020.git
    cd CLUENER2020
    git checkout master
    git reset --hard 2e8fccd1b8cde6b471fe440b9d766e920fb418ab
    export PYTHONPATH=`pwd`:$PYTHONPATH
    cd ..
    ```

## 准备数据集

1. 获取原始数据集  
    该模型使用CLUE_NER数据集，其数据是在清华大学开源的文本分类数据集THUCTC基础上，选出部分数据进行细粒度命名实体标注，原数据来源于Sina News RSS。详情可参考[https://www.cluebenchmarks.com/introduce.html](https://www.cluebenchmarks.com/introduce.html)。执行以下命令获取数据集：
    ```bash
    cd CLUENER2020/bilstm_crf_pytorch/
    python download_clue_data.py --data_dir=./dataset --tasks=cluener
    cd ../../
    ```

    执行上述命令后，生成的数据目录结构如下：
    ```
    ├── CLUENER2020/
        ├── bilstm_crf_pytorch/
            ├── dataset/
                ├── cluener/
                    ├── README.md
                    ├── cluener_predict.json
                    ├── dev.json
                    ├── test.json
                    └── train.json
    ```

2. 数据预处理  

    模型无法处理原始文本，对原始文本进行命名实体识别前，需要将原始文本转化为模型可识别的词向量，参考以下命令：
    ```bash
    python bilstm_preprocess.py \
        --data ./CLUENER2020/bilstm_crf_pytorch/dataset/cluener/dev.json \
        --max_seq_len ${max_seq_len} \
        --output inference/prep_data/
    ```
    + 参数说明：  
        + --data: 需要推理的文本文件。
        + --max_seq_len: 模型能够处理的最大文本长度，根据实际业务场景设置
        + --output: 预处理结果的保存目录

    预处理后的数据目录结构如下：
    ```
    ├── inference/
        └── prep_data/
            ├── inputs/
                ├── input_ids/
                    ├── data_00000000.npy
                    ├── ...
                    └── data_00001342.npy
                └── input_mask/
                    ├── data_00000000.npy
                    ├── ...
                    └── data_00001342.npy
            └── label.txt
    ```

## 模型转换

1. PyTroch 模型转 ONNX 模型  
 
    原仓未提供预训练权重，可参考原仓的文档自己训练，由于数据集较小，训练很快，整个训练过程在1小时左右：
    ```bash
    cd CLUENER2020/bilstm_crf_pytorch/
    # 在NPU机器上训练需将gpu设为空
    python run_lstm_crf.py --do_train --gpu ''
    cd ../../
    ```

    训练完成后，权重的保存路径默认为`outputs/bilstm_crf/best-model.bin`。然后执行`pth2onnx.py`，生成动态batchsize的ONNX模型：
    ```bash
    python bilstm_pth2onnx.py \
        --checkpoint CLUENER2020/bilstm_crf_pytorch/outputs/bilstm_crf/best-model.bin \
        --max_seq_len ${max_seq_len} \
        --output ./inference/bilstm_dybs.onnx
    ```
    + 参数说明：  
        + --checkpoint: 预训练权重文件路径
        + --max_seq_len: 模型能够处理的最大文本长度，根据实际业务场景设置
        + --output: 生成ONNX文件的保存路径。
    
    + 注：在线模型中的CRF模块实际上在模型forward执行后才被调用，属于后处理操作。因此，生成的ONNX模型中不包含CRF，而是将CRF模块放在了后处理`postprocess.py`中。但因为CRF模块中`transitions`的权重也存放在了预训练权重里，所以生成ONNX时，还附带将`transitions`权重值单独保存为`transitions.npy`，位于ONNX模型的同级目录下，推理结果的后处理将会用到此文件。


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
    bs=1  # 设置OM模型的batchsize
    
    # 执行 ATC 进行模型转换
    atc --framework=5 \
        --model=./inference/bilstm_dybs.onnx \
        --output=./inference/bilstm_bs${bs} \
        --input_format=ND \
        --input_shape="ids:${bs},${max_seq_len};mask:${bs},${max_seq_len}" \
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
    该离线模型使用ais_bench作为推理工具，请参考ais_bench的[**Gitee主页**](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)安装推理后端包aclruntime与推理前端包ais_bench。  
    可直接使用命令方式对预处理后的数据进行推理，参考命令：
    ```bash
    python -m ais_bench \
        --model ./inference/bilstm_bs${bs}.om \
        --input inference/prep_data/inputs/input_ids,inference/prep_data/inputs/input_mask \
        --output ./inference/ \
        --output_dirname results_bs${bs} \
        --outfmt NPY
    ```
    参数说明：
    + --model: OM模型路径
    + --input: 存放预处理后数据的目录路径
    + --output: 用于存放推理结果的父目录路径
    + --output_dirname: 用于存放推理结果的子目录名，位于--output指定的目录下
    + --outfmt: 推理结果的保存格式

2. 精度验证  

    使用后处理脚本计算模型的各精度指标：
    ```bash
    python bilstm_postprocess.py \
        --infer_results inference/results_bs1 \
        --annotations inference/prep_data/label.txt
    ```
    参数说明：
    + --infer_results: OM模型路径
    + --annotations: 存放预处理后数据的目录路径

    运行结束后，程序将会打印出模型整体精度指标以及在各个类别上的精度指标：
    ```
    metrics:
    {'acc': 0.7354972375690608, 'recall': 0.693359375, 'f1': 0.7138069705093834}
    metrics of per category:
    {
        "organization": {
            "acc": 0.771,
            "recall": 0.7248,
            "f1": 0.7472
        },
        "company": {
            "acc": 0.7459,
            "recall": 0.7143,
            "f1": 0.7297
        },
        ...
    }
    ```

3. 性能验证  
    对于性能的测试，需要注意以下三点：
    + 测试前，请通过`npu-smi info`命令查看NPU设备状态，请务必在NPU设备空闲的状态下进行性能测试。
    + 为了避免测试过程因持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps，反映模型在单位时间（1秒）内处理的样本数。
    ```bash
    python -m ais_bench --model ./inference/bilstm_bs${bs}.om --loop 100
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。


----
# 性能&精度

在310P设备上，OM模型精度为  **{acc=0.735, recall=0.693, f1=0.714}**，优于原仓提供的精度[f1=0.7](https://github.com/CLUEbenchmark/CLUENER2020/tree/master#%E6%95%88%E6%9E%9C%E5%AF%B9%E6%AF%94)，当batchsize设为32时OM模型性能最优，达 **961 fps**。

| 芯片型号   | BatchSize | 数据集   | 精度                               | 性能      |
| --------- | --------- | -------- | --------------------------------- | --------- |
|Ascend310P3| 1         | CLUE_NER | acc=0.735, recall=0.693, f1=0.714 | 60 fps    |
|Ascend310P3| 4         | CLUE_NER | acc=0.735, recall=0.693, f1=0.714 | 241 fps   |
|Ascend310P3| 8         | CLUE_NER | acc=0.735, recall=0.693, f1=0.714 | 479 fps   |
|Ascend310P3| 16        | CLUE_NER | acc=0.735, recall=0.693, f1=0.714 | 954 fps   |
|Ascend310P3| 32        | CLUE_NER | acc=0.735, recall=0.693, f1=0.714 | **961 fps** |
|Ascend310P3| 64        | CLUE_NER | acc=0.735, recall=0.693, f1=0.714 | 957 fps   |
