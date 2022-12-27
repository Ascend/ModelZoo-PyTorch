# Rosetta_MobileNetV3 模型推理指导

- [Rosetta\_MobileNetV3 模型推理指导](#rosetta_mobilenetv3-模型推理指导)
- [概述](#概述)
- [推理环境](#推理环境)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型转换](#模型转换)
  - [推理验证](#推理验证)
- [性能\&精度](#性能精度)

----
# 概述

Rosetta是用于图像中文本检测和识别的大规模系统，文本识别是使用称为 CTC 的全卷积模型完成的（因为它在训练期间使用序列到序列的 CTC 损失），该模型输出字符序列。最后一个卷积层在输入词的每个图像位置预测最可能的字符。


+ 论文  
    [Rosetta: Large scale system for text detection and recognition in images](https://arxiv.org/pdf/1910.05085.pdf)  
    [Fedor Borisyuk](https://arxiv.org/search/cs?searchtype=author&query=Borisyuk%2C+F), [Albert Gordo](https://arxiv.org/search/cs?searchtype=author&query=Gordo%2C+A), [Viswanath Sivakumar](https://arxiv.org/search/cs?searchtype=author&query=Sivakumar%2C+V)

+ 参考实现  
    ```
    url = https://github.com/PaddlePaddle/PaddleOCR.git
    branch = release/2.6
    commit_id = 76fefd56f86f2809a6e8f719220746442fbab9f3
    model_name = Rosetta_MobileNetV3
    ```

+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | x      |  RGB_FP32 | NCHW        | batchsize x 3 x 32 x 100 |

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | softmax_0.tmp_0 |  FLOAT32   | ND          | batchsize x 25 x 37 |



----
# 推理环境

- 该模型推理所需配套的软件如下：

    | 配套      | 版本    | 环境准备指导 |
    | --------- | ------- | ---------- |
    | 固件与驱动 | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN      | 5.1.RC2 | -          |
    | Python    | 3.7.5   | -          |
    
    说明：该模型离线推理使用 Atlas 300I Pro 推理卡，Atlas 300I Duo 推理卡请以 CANN 版本选择实际固件与驱动版本。


----
# 快速上手

## 获取源码

1. 克隆开源仓源码。
    ```shell
    git clone -b release/2.6 https://github.com/PaddlePaddle/PaddleOCR.git
    cd PaddleOCR
    git reset --hard 76fefd56f86f2809a6e8f719220746442fbab9f3
    ```

2. 下载本仓，将该模型目录下的Python脚本、requirements.txt与补丁文件复制到当前目录，并修改源码。
    ```shell
    patch -p1 < rosetta.patch
    ```

3. 执行以下命令安装所需的依赖。
    ```shell
    pip install -r requirements.txt
    python setup.py install
    ```

4. 创建一个目录，用于存放整个推理过程中所需文件与生成文件。
    ```shell
    mkdir rosetta
    ```

## 准备数据集

1. 获取原始数据集  
    该模型在以 LMDB 格式(LMDBDataSet)存储的 IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE 数据集上进行评估，共计 12067 个评估数据，数据介绍参考 [DTRB](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fclovaai%2Fdeep-text-recognition-benchmark%23download-lmdb-dataset-for-traininig-and-evaluation-from-here)。请自行下载[数据集](https://gitee.com/link?target=https%3A%2F%2Fwww.dropbox.com%2Fsh%2Fi39abvnefllx2si%2FAAAbAYRvxzRp3cIE5HzqUw3ra%3Fdl%3D0)，并解压到 rosetta/lmdb/ 目录下，解压好的数据目录结构如下：
    ```
    rosetta/lmdb/
    ├── CUTE80/
        └── data.mdb
        └── lock.mdb
    ├── IC03_860/
        ├── data.mdb
        └── lock.mdb
    ├── IC03_867/
        ├── data.mdb
        └── lock.mdb
    ├── IC13_1015/
        ├── data.mdb
        └── lock.mdb
    ├── IC13_857/
        ├── data.mdb
        └── lock.mdb
    ├── IC15_1811/
        ├── data.mdb
        └── lock.mdb
    ├── IC15_2077/
        ├── data.mdb
        └── lock.mdb
    ├── IIIT5k_3000/
        ├── data.mdb
        └── lock.mdb
    ├── SVT/
        ├── data.mdb
        └── lock.mdb
    └── SVTP/
        ├── data.mdb
        └── lock.mdb
    ```

2. 数据预处理  
    执行前处理脚本将原始数据转换为 OM 模型输入需要的 bin 文件。
    ```shell
    python rosetta_preprocess.py \
        --config configs/rec/rec_mv3_none_none_ctc.yml \
        --opt data_dir=rosetta/lmdb/ bin_dir=rosetta/bin_list info_dir=rosetta/info_list
    ```
    参数说明：
    + -c, --config: 模型配置文件路径
    + --opt data_dir: lmdb数据集所在路径
    + --opt bin_dir: 存放生成的bin文件的目录路径
    + --opt info_dir: 存放groundtruth等信息的目录路径
    
    预处理脚本运行结束后，rosetta/bin_list 目录下会生成 12067 个 bin 文件。 rosetta/info_list 目录下也会对应生成 12067 个 pickle 文件


## 模型转换

1. PyTroch 模型转 ONNX 模型  

    step1: 下载 paddle 预训练模型
    下载 PaddleOCR 提供的 [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_none_ctc_v2.0_train.tar) 到 rosetta 目录下，然后解压。
    ```shell
    cd rosetta
    wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_none_ctc_v2.0_train.tar
    tar xf rec_mv3_none_none_ctc_v2.0_train.tar
    cd ..
    ```

    step2: paddle 训练模型转 paddle 推理模型
    ```
    python3 tools/export_model.py \
        -c configs/rec/rec_mv3_none_none_ctc.yml \
        -o Global.pretrained_model=rosetta/rec_mv3_none_none_ctc_v2.0_train/best_accuracy \
           Global.save_inference_dir=rosetta/rec_mv3_none_none_ctc_v2.0_infer/
    ```
    参数说明：
    + -c, --config: paddle模型配置文件路径
    + -o Global.pretrained_model: paddle预训练模型路径
    + -o Global.save_inference_dir: paddle推理模型保存目录

    step3: 生成 ONNX 模型
    ```shell
    paddle2onnx \
        --model_dir rosetta/rec_mv3_none_none_ctc_v2.0_infer/ \
        --model_filename inference.pdmodel \
        --params_filename inference.pdiparams \
        --save_file rosetta/rosetta_mobilenetv3.onnx \
        --opset_version 11 \
        --input_shape_dict="{'x':[-1,3,-1,-1]}" \
        --enable_onnx_checker True
    ```
    参数说明：
    + --model_dir: paddle推理模型所在的目录路径
    + --save_file: 生成ONNX模型的保存路径
    + --input_shape_dict: PaddleOCR模型转化过程中必须采用动态shape的形式，所以此处固定设为{'x':[-1,3,-1,-1]}
    + --opset_version: ONNX算子集版本
    
    注：PaddleOCR模型转ONNX，详情请参考 [Paddle2ONNX模型转化与预测](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/deploy/paddle2onnx/readme.md)。

2. ONNX 模型转 OM 模型  

    step1: 查看NPU芯片名称 \${chip_name}
    ```shell
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
    ```shell
    # 配置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /etc/profile
    
    # 执行 ATC 进行模型转换
    atc --framework=5 \
        --model=rosetta/rosetta_mobilenetv3.onnx \
        --input_shape="x:${bs},3,32,100" \
        --output=rosetta/rosetta_mobilenetv3_bs${bs} \
        --input_format=NCHW \
        --log=error \
        --soc_version=Ascend${chip_name}
    ```

   参数说明：
    + --model: 为ONNX模型文件。
    + --framework: 5代表ONNX模型。
    + --input_shape: 输入数据的shape。
    + --input_format: 输入数据的排布格式。
    + --output: 输出的OM模型。
    + --log：日志级别。
    + --soc_version: 处理器型号。

    命令中的`${bs}`表示模型输入的 batchsize，比如将`${bs}`设为 1，则运行结束后会在 rosetta 目录下生成 rosetta_mobilenetv3_bs1.om


## 推理验证

1. 准备推理工具  

    本推理项目使用 ais_bench 作为推理工具，须自己打包并安装。
    请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

2. 离线推理  

    使用ais_bench推理工具将预处理后的数据传入模型并执行推理：
    ```shell
    # 设置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /etc/profile
    
    # 对预处理后的数据进行推理
    mkdir rosetta/val_bs1_out/
    python3 -m ais_bench --model rosetta/rosetta_mobilenetv3_bs1.om --input rosetta/bin_list/ --output rosetta/val_bs1_out/ --batchsize 1
    ```
    参数说明：
    + --model: OM模型路径。
    + --input: 存放预处理bin文件的目录路径
    + --output: 存放推理结果的目录路径
    + --batchsize: 模型每次处理样本的数量
    
    运行成功后，在 rosetta/val_bs1_out/ 下，会生成一个以执行开始时间`%Y_%m_%d-%H_%M_%S`来命名的子目录，每个预处理 bin 文件会对应生成一个推理结果 bin 文件存放在此目录下。
  
3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的准确率：
    ```shell
    python rosetta_postprocess.py \
        -c configs/rec/rec_mv3_none_none_ctc.yml \
        -o res_dir=rosetta/val_bs1_out/2022_09_26-15_13_53/ info_dir=rosetta/info_list/
    ```
    参数说明：
    + -c, --config: 模型配置文件路径
    + -o res_dir: 存放推理结果的目录路径
    + -o info_dir: 存放groundtruth等信息的目录路径
    
    运行成功后，程序会打印出模型的精度指标：
    ```
    acc: 0.773846025711572
    norm_edit_dis: 0.9074651623274426
    ```

4. 性能验证  

    对于性能的测试，需要注意以下三点：
    + 测试前，请通过 npu-smi info  命令查看 NPU 设备状态，请务必在 NPU 设备空闲的状态下进行性能测试。
    + 为避免因测试持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps.

    > 吞吐率（throughput）：模型在单位时间（1秒）内处理的数据样本数。
    
    执行纯推理：
    ```shell
    python3 -m ais_bench --model rosetta/rosetta_mobilenetv3_bs1.om --loop 100 --batchsize 1
    ```

    执行完纯推理命令，程序会打印出与性能相关的指标：
    ```
    [INFO] -----------------Performance Summary------------------
    [INFO] H2D_latency (ms): min = 0.10275840759277344, max = 0.10275840759277344, mean = 0.10275840759277344, median = 0.10275840759277344, percentile(99%) = 0.10275840759277344
    [INFO] NPU_compute_time (ms): min = 0.40299999713897705, max = 3.2839999198913574, mean = 0.4343369999527931, median = 0.42399999499320984, percentile(99%) = 0.5660099846124649
    [INFO] D2H_latency (ms): min = 0.07843971252441406, max = 0.07843971252441406, mean = 0.07843971252441406, median = 0.07843971252441406, percentile(99%) = 0.07843971252441406
    [INFO] throughput 1000*batchsize(1)/NPU_compute_time.mean(0.4343369999527931): 2302.359688694924
    [INFO] ------------------------------------------------------

    ```
    
    计算吞吐率：
    + 执行纯推理时若指定了 batchsize，则找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率，本例中的吞吐率为 2302.35 fps.
    + 若没有指定 batchsize，则可以通过 **NPU_compute_time** 中的 **mean** 来计算：
    $$throughput =\frac{batchsize}{mean} * 1000 = 2302.36(fps)$$

----
# 性能&精度

1. 性能对比
  
    在 310P 设备上，当 batchsize 为 64 时模型性能最优，达 24219 fps.
    | batchsize | T4性能 | 310P性能 | 310P/T4 |
    | --- | -------- | ---- | ---- |
    | 1   | 1828 fps | 2302 fps | 1.26倍 |
    | 4   | 6489 fps | 6799 fps | 1.05倍 |
    | 8   | 10259 fps | 13743 fps | 1.34倍 | 
    | 16  | 13547 fps | 18703 fps | 1.38倍 | 
    | 32  | 14373 fps | 22191 fps | 1.54倍 | 
    | 64  | 14985 fps | 24219 fps | 1.62倍 | 
    |best | **14985 fps** | **24219 fps** | **1.62倍** |

2. 精度对比

    在 310P 设备上，OM 模型各个 batchsize 的精度均为 77.38%，优于 PaddleOCR 官方提供的指标
    | 模型名 | Avg-Acc(310P实测) | Avg-Acc(PaddleOCR) |
    |-------|-------------------|-----------------|
    | Rosetta_MobileNetV3 | 77.38% | [75.80%](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_ch/algorithm_rec_rosetta.md#1-%E7%AE%97%E6%B3%95%E7%AE%80%E4%BB%8B) |
