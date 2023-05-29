#  SSD-MobileNetV2 模型-推理指导


- [概述](#概述)

- [推理环境准备](#推理环境准备)

- [快速上手](#快速上手)

  + [获取源码](#获取源码)
  + [准备数据集](#准备数据集)
  + [模型转换](#模型转换)
  + [推理验证](#推理验证)

- [精度&性能](#精度性能)


# 概述

SSD-MobileNetV2 采用 SSD 的思想，在MobileNetV2基础上，中间层提取了一些featuremap。

- 参考实现：

    ```
    url = https://github.com/qfgaohao/pytorch-ssd.git
    branch = master
    commit_id = f61ab424d09bf3d4bb3925693579ac0a92541b0d 
    ```

## 输入输出数据

- 输入数据

    | 输入数据  | 数据类型 | 大小                      | 数据排布格式   |
    | -------- | -------- | ------------------------- | ------------ |
    | image    | RGB_FP32 | batchsize x 3 x 300 x 300 | NCHW         |

- 输出数据

    | 输出数据  | 大小                  | 数据类型  | 数据排布格式  |
    | -------- | --------------------- | -------- | ------------ |
    | scores   | batchsize x 3000 x 21 | FLOAT32  | ND           |
    | boxes    | batchsize x 3000 x 4  | FLOAT32  | ND           |



# 推理环境准备

- 该模型需要以下插件与驱动

    **表 1**  版本配套表
    
    | 配套        | 版本    | 环境准备指导 |
    | ----------- | ------- | ---------- |
    | 固件与驱动   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN        | 6.0.RC1 | -          |
    | Python      | 3.7.5   | -          |

    说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。

# 快速上手

## 获取源码

1. 获取源码。

    ```bash
    git clone https://github.com/qfgaohao/pytorch-ssd.git -b master
    cd pytorch-ssd
    git reset f61ab424d09bf3d4bb3925693579ac0a92541b0d --hard
    cd ..
    ```

2. 安装依赖。

    ```bash
    pip install -r requirements.txt
    ```

## 准备数据集

1. 获取原始数据集

    使用[**VOC2007的测试集**](https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar)作为测试数据集，参考以下命令下载测试图片与标签，并解压：
    ```bash
    wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
    tar -xvf VOCtest_06-Nov-2007.tar
    wget https://storage.googleapis.com/models-hao/voc-model-labels.txt -P VOCdevkit/
    ```
    
    或自行下载解压，按照以下的目录结构存放图片与标签即可：
    ```bash
    ./VOCdevkit/
        ├── VOC2007/
            ├── Annotations/
            ├── ImageSets/
            ├── JPEGImages/
            ├── SegmentationClass/
            └── SegmentationObject/
        └── voc-model-labels.txt
    ```

2. 数据预处理

    数据预处理将原始数据集转换为模型输入的数据。

    执行前处理脚本将原始图片转换为OM模型输入需要的bin文件。
    ```bash
    python ssdmobilenetv2_preprocess.py --src_path ./VOCdevkit/VOC2007/JPEGImages/ --save_path ./pre_dataset
    ```
    参数说明：
    + --src_path: 测试图片所在的目录路径
    + --save_path: 存放生成bin文件的目录路径

## 模型转换

1. 获取权重文件
    ```bash
    wget https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth
    ```

2. 导出onnx文件
    ```bash
    python ssdmobilenetv2_pth2onnx.py --ckpt mb2-ssd-lite-mp-0_686.pth --onnx mb2-ssd.onnx
    ```
    参数说明：
    + --ckpt: 预训练权重文件的路径
    + --onnx: 生成ONNX模型的保存路径

3. 使用ATC工具将ONNX模型转OM模型。
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
    bs=4  # 根据需要自行设置batchsize

    # 执行 ATC 进行模型转换
    atc --framework=5 \
        --model=mb2-ssd.onnx \
        --output=mb2-ssd_bs${bs} \
        --input_format=NCHW \
        --input_shape="image:${bs},3,300,300" \
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
    安装ais_bench推理工具。请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。完成安装后，执行以下命令预处理后的数据进行推理。
    ```bash
    python -m ais_bench \
        --model mb2-ssd_bs${bs}.om \
        --input ./pre_dataset/ \
        --output ./ \
        --output_dirname ./result_bs${bs}/ \
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
    python -m ais_bench --model mb2-ssd_bs${bs}.om --batchsize ${bs} --loop 100
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python ssdmobilenetv2_postprocess.py \
        --data_root ./VOCdevkit/VOC2007/ \
        --label_file ./VOCdevkit/voc-model-labels.txt \
        --infer_result ./result_bs${bs}/ \
        --eval_output ./metrics_bs${bs}/ 
    ```
    参数说明：
    + --data_root: 原始数据集路径
    + --label_file: 标签文件路径
    + --infer_results: 存放推理结果的目录路径
    + --eval_output: 指定一个目录用于存放模型的精度指标。
    
    运行成功后，模型在各个类别上的精度指标以及平均精度都可在`./metrics_bs${bs}/`目录下查看。


# 性能&精度

在310P设备上，各batchsize的OM模型在各个类别上的平均精度为精度为  **0.698**，当batchsize设为4时模型性能最优，达 2923 fps。

| 芯片型号   | BatchSize | 数据集      | 精度            | 性能       |
| --------- | --------- | ----------- | -------------- | ---------- |
|Ascend310P3| 1         | VOC2007     | 0.698          | 1511 fps   |
|Ascend310P3| 4         | VOC2007     | 0.698          | 2923 fps   |
|Ascend310P3| 8         | VOC2007     | 0.698          | 2841 fps   |
|Ascend310P3| 16        | VOC2007     | 0.698          | 2760 fps   |
|Ascend310P3| 32        | VOC2007     | 0.698          | 2601 fps   |
|Ascend310P3| 64        | VOC2007     | 0.698          | 2452 fps   |

