# Cross-Scale-Non-Local-Attention 模型推理指导

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

在自然图像中，跨尺度的图像相似性是普遍的，本文使用跨尺度的Non-Local注意力模型，有效挖掘图像内部先验知识，在多个实验中证明所提出的方法在多个SISR基准测试中取得了最先进的性能。

+ 论文  
    [Image Super-Resolution with Cross-Scale Non-Local Attention and Exhaustive Self-Exemplars Mining](https://arxiv.org/pdf/2111.06377.pdf)  
    Yiqun Mei, Yuchen Fan, Yuqian Zhou, Lichao Huang, Thomas S. Huang, Humphrey Shi

+ 参考实现：  
    https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention.git

## 输入输出数据
+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | input.1| FLOAT32 | NCHW | batch_size x 3 x 56 x 56 | 

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output1      |  FLOAT32   | NCHW | batch_size x 3 x 224 x 224        |


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
    ```bash
    git clone https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention.git -b master
    cd Cross-Scale-Non-Local-Attention/
    git reset af168f99afad2d9a04a3e4038522987a5a975e86 --hard
    cd ../
    ```
## 准备数据集

1. 获取原始数据集  
    本模型推理项目使用 Set5 数据集验证模型精度，请点击 [**set5**](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) 自行下载，并按照以下的目录结构存放图片与标签文件。   
    ```
    Cross-Scale-Non-Local-Attention
    ├── Set5
    │   ├── HR
    │   ├── LR_bicubic
    │   │   ├── X2
    │   │   ├── X3
    │   │   ├── X4
    ```


2. 数据预处理  
    执行前处理脚本将原始数据转换为OM模型输入需要的bin/npy文件。
    ```bash
    python3 CSNLN_preprocess.py --s  ./Set5/LR_bicubic/X4/ --d prep_dataset
    ```
    其中"s"表示处理前原数据集的地址，"d"表示生成数据集的文件夹名称



## 模型转换

1. PyTroch 模型转 ONNX 模型  

    获取权重文件[model_x4.pt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Cross-Scale-Non-Local-Attention/PTH/model_x4.pt)
 
    然后执行执行以下命令生成 ONNX 模型：
    ```
    python3 CSNLN_pth2onnx.py --n_feats 128 --pre_train model_x4.pt --save csnln_x4.onnx
    ```
    参数说明：
     + --n_feats : 特征大小。
     + --pre_train : 权重文件路径。
     + --save : onnx保存路径。

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
    batch_size=1  # 根据需要自行设置 
  
    # 执行 ATC 进行模型转换
    atc --model=./csnln_x4_perf.onnx \
        --framework=5 \
        --output=csnln_x4_bs1 \
        --input_format=NCHW \
        --input_shape="input.1:1,3,56,56" \
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
        --model csnln_x4_bs1.om \
        --input ./prep_dataset/bin_56 \ 
        --output ./result/ \
        --outfmt BIN \
        --batchsize 1
    ```
    参数说明：
    + --model OM模型路径
    + --input 存放预处理后数据的目录路径
    + --output 用于存放推理结果的父目录路径
    + --outfmt 推理结果文件的保存格式
    + --batchsize 模型每次输入bin文件的数量

2. 性能验证  
    对于性能的测试，需要注意以下三点：
    + 测试前，请通过`npu-smi info`命令查看NPU设备状态，请务必在NPU设备空闲的状态下进行性能测试。
    + 为了避免测试过程因持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps，反映模型在单位时间（1秒）内处理的样本数。
    ```bash
    python3 -m ais_bench --model csnln_x4_bs1.om --batchsize 1
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python3 CSNLN_postprocess.py --hr  ./Set5/HR/ --res result/dumpOutput_device0 --save_path res_png
    ```
    参数说明：
    + --hr：生成推理结果所在路径。
    + --res：标签数据。
    + --save_path：生成结果文件。
    + --json-file-name: 精度文件名。
    + --batch-size: 输入文件数量。
    

----
# 性能&精度

在310P设备上，OM模型的精度为  

| 芯片型号   | BatchSize | 数据集      | 精度            | 性能       |
| --------- | --------- | ----------- | --------------- | --------- |
|Ascend310P3| 1         | Set5  | 32.57 | 0.7163 fps |

备注：由于内存限制，离线模型不支持多batch