# PoseC3D 模型推理指导

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

PoseC3D 是一种基于 3D-CNN 的骨骼行为识别框架，同时具备良好的识别精度与效率，在包含 FineGYM, NTURGB+D, Kinetics-skeleton 等多个骨骼行为数据集上达到了 SOTA。不同于传统的基于人体 3 维骨架的 GCN 方法，PoseC3D 仅使用 2 维人体骨架热图堆叠作为输入，就能达到更好的识别效果。

+ 论文  
    [Revisiting Skeleton-based Action Recognition](https://arxiv.org/abs/2104.13586)  
    Haodong Duan, Yue Zhao, Kai Chen, Dahua Lin, Bo Dai  

+ 参考实现：  
    url = https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/posec3d  
    tag = v0.24.1  
    config = slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint  

## 输入输出数据
+ 模型输入  
    | 输入数据         | 数据类型   | 大小                        | 数据排布格式 |
    | --------------- | --------- | --------------------------- | ----------- |
    | onnx::Reshape_0 | FLOAT32   | bs x 20 x 17 x 48 x 56 x 56 | ND          | 

+ 模型输出  
    | 输出数据 |  数据类型 | 大小          | 数据排布格式 |
    | ------- | --------- | ------------ | ----------- |
    | -       |  FLOAT32  | (bs*20) x 51 | ND          |


----
# 推理环境

- 该模型推理所需配套的软件如下：

    | 配套      | 版本    | 环境准备指导 |
    | --------- | ------- | ---------- |
    | 固件与驱动 | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN      | 6.0.RC1 | -          |
    | Python    | 3.8.13  | -          |
    
    说明：请根据推理卡型号与 CANN 版本选择相匹配的固件与驱动版本。


----
# 快速上手

## 获取源码

1. 安装推理过程所需的依赖
    ```bash
    pip install -r requirements.txt
    ```
2. 获取开源仓源码
    ```bash
    git clone -b v0.24.1 https://github.com/open-mmlab/mmaction2.git
    pip install -v -e mmaction2
    ```

## 准备数据集

1. 获取原始数据集  
    该模型使用[HMDB51数据集](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)来验证精度，参考[mmaction2官方提供的hmdb51数据集获取与处理方法](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/hmdb51)，下载原始视频并截帧，最后生成文件列表。
    ```bash
    cd ./mmaction2/tools/data/hmdb51
    bash download_videos.sh
    bash extract_rgb_frames_opencv.sh
    cd -
    wget https://download.openmmlab.com/mmaction/posec3d/hmdb51.pkl -P mmaction2/data/hmdb51/
    rm -rf mmaction2/data/hmdb51/videos
    ```
    执行上述命令后，生成的数据目录结构如下：
    ```
    ├── mmaction2/
        ├── data/
            ├── hmdb51/
                ├── hmdb51.pkl
                └── rawframes/
                    ├── brush_hair/
                    ├── cartwheel/
                    ├── ...
                    ├── walk/
                    └── wave/
    ```


2. 数据预处理  
    执行前处理脚本将原始数据转换为推理工具支持的bin文件。
    ```bash
    python posec3d_preprocess.py \
        --frame_dir ./mmaction2/data/hmdb51/rawframes/ \
        --ann_file ./mmaction2/data/hmdb51/hmdb51.pkl \
        --output_dir ./prep_data
    ```
    参数说明：
    + --frame_dir: 视频截帧后的存放目录
    + --ann_file: 标注文件路径
    + --output_dir: 预处理结果的保存目录
    
    执行上述命令后，`./prep_data`目录下会生成一个名为`bin`的子目录，存放生成的1530个bin文件，此外还会生成`./prep_data/hmdb51_label.txt`文件，其内容为预处理后的每个bin文件对应的标签。

## 模型转换

1. PyTroch 模型转 ONNX 模型  
 
    下载open-mmlab官方提供的[**预训练模型**](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint-76ffdd8b.pth)到当前目录，可参考命令：
    ```bash
    wget https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint-76ffdd8b.pth
    ```

    执行mmaction官方提供的转ONNX脚本，生成ONNX模型：
    ```bash
    python mmaction2/tools/deployment/pytorch2onnx.py \
        ./mmaction2/configs/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint.py \
        ./slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint-76ffdd8b.pth \
        --shape ${bs} 20 17 48 56 56 \
        --output-file ./posec3d_bs${bs}.onnx
    ```
    说明：前两个位置参数分别为模型配置文件路径与预训练权重文件路径；`--shape`为模型输入的shape，可对bs设置不同值以生成不同batchsize的ONNX模型；`--output-file`为生成ONNX模型的保存路径。


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
    atc --framework=5 \
        --model=./posec3d_bs${bs}.onnx \
        --output=./posec3d_bs${bs} \
        --input_format=ND \
        --input_shape="onnx::Reshape_0:${bs},20,17,48,56,56" \
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
    该离线模型使用ais_infer作为推理工具，请参考[**安装文档**](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench#%E4%B8%80%E9%94%AE%E5%AE%89%E8%A3%85)安装推理后端包aclruntime与推理前端包ais_bench。完成安装后，执行以下命令预处理后的数据进行推理。
    ```bash
    python -m ais_bench \
        --model ./posec3d_bs${bs}.om \
        --input ./prep_data/bin/ \
        --output ./ \
        --output_dirname result_bs${bs} \
        --batchsize ${bs}
    ```
    参数说明：
    + --model OM模型路径
    + --input 存放预处理后数据的目录路径
    + --output 用于存放推理结果的父目录路径
    + --output_dirname 用于存放推理结果的子目录名，位于--output指定的目录下
    + --batchsize 模型每次输入bin文件的数量


2. 性能验证  
    对于性能的测试，需要注意以下三点：
    + 测试前，请通过`npu-smi info`命令查看NPU设备状态，请务必在NPU设备空闲的状态下进行性能测试。
    + 为了避免测试过程因持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps，反映模型在单位时间（1秒）内处理的样本数。
    ```bash
    python -m ais_bench --model ./posec3d_bs${bs}.om --batchsize ${bs} --loop 100
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的精度指标：
    ```bash
    python posec3d_postprocess.py \
        --infer_results ./result_bs${bs} \
        --label_file ./prep_data/hmdb51_label.txt
    ```
    参数说明：
    + --infer_results: 推理结果所在路径
    + --label_file: 预处理后的标签文件路径
    
    运行结束后，程序会打印出OM模型的精度指标：
    ```
    Evaluating top_k_accuracy ...
    top1_acc        0.6922
    top5_acc        0.9131
    ```

----
# 性能&精度

在310P设备上，OM模型精度为  **{Top1@Acc=69.22%, Top5@Acc=91.31%}**，当batchsize设为8时OM模型性能最优，达 **22.39 fps**。

| 芯片型号   | BatchSize | 数据集       | 精度            | 性能       |
| --------- | --------- | ------------ | --------------- | --------- |
|Ascend310P3| 1         | HMDB51       | Top1@Acc=69.22%, Top5@Acc=91.31 | 22.05 fps |
|Ascend310P3| 4         | HMDB51       | Top1@Acc=69.22%, Top5@Acc=91.31 | 22.16 fps |
|Ascend310P3| 8         | HMDB51       | Top1@Acc=69.22%, Top5@Acc=91.31 | **22.39 fps** |

说明：在310P服务器上，当batchsize为16或更高时，OM模型因内存不足无法推理。
