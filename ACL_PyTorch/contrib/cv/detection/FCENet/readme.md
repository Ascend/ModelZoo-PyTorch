#  FCENet 模型推理指导

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

FCENet，使用傅里叶变换来得到文本的包围框，该方法在弯曲文本数据集（ctw1500、total-text）上可以达到SOTA的效果

+ 论文  
    [FCENet论文](https://arxiv.org/abs/2104.10442)
    Yiqin Zhu, Jianyong Chen, Lingyu Liang, Zhanghui Kuang, Lianwen Jin, Wayne Zhang

+ 参考实现：
    ```  
    https://github.com/open-mmlab/mmocr.git
    branch:master
    ```
## 输入输出数据
+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | actual_input_1 | FLOAT32 | NCHW | bs x 3 x 1280 x 2272 | 

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output1      |  FLOAT32   | ND          | bs x 4 x 160 x 284     |
    | output2      |  FLOAT32   | ND          | bs x 22 x 160 x 284     |
    | output3      |  FLOAT32   | ND          | bs x 4 x 80 x 142     |
    | output4      |  FLOAT32   | ND          | bs x 22 x 80 x 142     |
    | output5      |  FLOAT32   | ND          | bs x 4 x 40 x 71     |
    | output6      |  FLOAT32   | ND          | bs x 22 x 40 x 71     |

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
    git clone https://github.com/open-mmlab/mmocr.git
    cd mmocr
    git reset --hard 662f87106cfeca636523787b4b17a8e8967edc1c
    patch -p1 < ../fcenet.patch
    pip3 install -r requirements.txt
    pip3 install -v -e . # or "python setup.py develop"
    export PYTHONPATH=$(pwd):$PYTHONPATH
    cd ..
    ```
## 准备数据集

1. 获取原始数据集  
    本模型需要icdar2015数据集，数据集请参考开源代码仓方式获取。获取icdar2015数据集，放到mmocr的data文件夹内,放置顺序如下。
    ```
    ├── icdar2015
    │   ├── imgs
    │   ├── annotations
    │   ├── instances_test.json
    │   └── instances_training.json
    ```

2. 数据预处理  
    执行前处理脚本将原始数据转换为OM模型输入需要的bin/npy文件。
    ```
    mkdir preprocessed_imgs
    python3 fcenet_preprocess.py ./mmocr/data/icdar2015/imgs/test/
    ```



## 模型转换


1. 下载pth权重文件  
[FCENet预训练pth权重文件](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/detection/FCENet/fcenet_r50_fpn_1500e_icdar2015_20211022-daefb6ed.pth)  
我们利用官方的PTH文件进行验证，官方PTH文件可从原始开源库中获取，我们需要fcenet_r50_fpn_1500e_icdar2015_20211022-daefb6ed.pth文件，并放在当前工作目录下。

    然后执行执行以下命令生成 ONNX 模型：
    ```bash
    python3 ./pytorch2onnx.py 
    ./mmocr/configs/textdet/fcenet/fcenet_r50_fpn_1500e_icdar2015.py \
    ./fcenet_r50_fpn_1500e_icdar2015_20211022-daefb6ed.pth \
    det \
    ./mmocr/data/icdar2015/imgs/test/img_1.jpg \
    --dynamic-export \
    --output-file ./fcenet_dynamicbs.onnx

    ```
    参数说明：
    + --dynamic-export: 是否动态导出onnx模型
    + --output-file: 输出onnx的文件名。


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
    batch_size=1
    
    # 执行 ATC 进行模型转换
    atc --model=./fcenet_dynamicbs.onnx \
        --framework=5 \
        --output=./fcenet_bs${batch_size}  \
        --input_format=NCHW \
        --input_shape="input:$batch_size,3,1280,2272" \
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
    python3 -m ais_bench \
        --model ./fcenet_bs${batch_size} \
        --input ./preprocessed_imgs/ \ 
        --output ./result \
        --outfmt TXT \
        --batchsize ${batch_size}
    ```
    参数说明：
    + --model OM模型路径
    + --input 存放预处理后数据的目录路径
    + --output 用于存放推理结果的父目录路径
    + --outfmt 推理结果文件的保存格式
    + --batchsize 模型每次输入bin文件的数量,本例中为1。


2. 性能验证  
    对于性能的测试，需要注意以下三点：
    + 测试前，请通过`npu-smi info`命令查看NPU设备状态，请务必在NPU设备空闲的状态下进行性能测试。
    + 为了避免测试过程因持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps，反映模型在单位时间（1秒）内处理的样本数。
    ```bash
    python3 -m ais_bench --model ./fcenet_bs${batch_size} --batchsize ${batch_size}
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python3 fcenet_postprocess.py \
            --input_path=./result \
            --instance_file=./mmocr/data/icdar2015/instances_test.json \
        --output_file=./boundary_results.txt
    python3 eval.py \
    ./mmocr/configs/textdet/fcenet/fcenet_r50_fpn_1500e_icdar2015.py \
    fcenet_r50_fpn_1500e_icdar2015_20211022-daefb6ed.pth \
    --eval hmean-iou
    ```
    参数说明：
    + --input_path: 存放推理结果的目录路径
    + --instance_file: 标签文件路径
    + --output_file: 输出结果文件名

    ```bash
    python3 eval.py \
    ./mmocr/configs/textdet/fcenet/fcenet_r50_fpn_1500e_icdar2015.py \
    fcenet_r50_fpn_1500e_icdar2015_20211022-daefb6ed.pth \
    --eval hmean-iou
    ```
    参数说明：
    + --eval : 策略模式



----
# 性能&精度

在310P设备上，OM模型的精度为  **{acc=0.872}**，当batchsize设为1时模型性能最优，达 28.9 fps。

| 芯片型号   | BatchSize | 数据集      | 精度            | 性能       |
| --------- | --------- | ----------- | --------------- | --------- |
|Ascend310P3| 1         | icdar2015  | acc=0.872| 28.9 fps |
|Ascend310P3| 4         | icdar2015  | acc=0.872| 26.8 fps |
|Ascend310P3| 8         | icdar2015  | acc=0.872| 20.7 fps |
|Ascend310P3| 16        | icdar2015  | acc=0.872| 20.7 fps |
|Ascend310P3| 32        | icdar2015  | acc=0.872| 20.8 fps |
|Ascend310P3| 64        | icdar2015  | acc=0.872| 20.8 fps |

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md