
# Ultra-Fast-Lane-Detection 模型推理指导

- [Ultra-Fast-Lane-Detection 模型推理指导](#ultra-fast-lane-detection-模型推理指导)
- [概述](#概述)
- [推理环境](#推理环境)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型转换](#模型转换)
  - [推理验证](#推理验证)
- [精度\&性能](#精度性能)

----
# 概述

现行主流的车道检测方法将车道检测视为像素分割问题，难以解决高挑战性的场景与速度问题。受人类感知的启发，严重遮挡和极端光照条件下的车道识别主要基于上下文和全局信息。由此，作者针对极快的速度和复杂的场景提出了一种新颖、简单但有效的方法。具体来说，作者将车道检测过程视为使用全局特征的 **row-based selecting**，这使得计算成本显著降低。使用全局特征的大感受野，也可以处理高挑战性的场景。此外，作者还提出了 **structural loss** 用以显式地模拟车道结构。在两个车道检测基准数据集上进行的大量实验表明，此方法在速度和准确性方面都可以达到目前的最佳水平。轻量级版本甚至可以在相同分辨率下达到每秒 300+ 帧，这比以前最先进的方法至少快 4 倍。

+ 论文  
    [Ultra Fast Structure-aware Deep Lane Detection](https://arxiv.org/abs/2004.11757)
    
+ 参考实现  
    ```
    url = https://github.com/cfzd/Ultra-Fast-Lane-Detection.git
    branch = master
    commit_id = f58fcd5f58511159ebfd06e60c7e221558075703
    ```

+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | input      |  RGB_FP32 | NCHW        | batchsize x 3 x 288 x 800 |

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output      |  FLOAT32   | ND          | batchsize x 101 x 56 x 4 |



----
# 推理环境

- 该模型离线推理使用 Atlas 300I Pro 推理卡，推理所需配套的软件如下：

    | 配套      | 版本    | 环境准备指导 |
    | --------- | ------- | ---------- |
    | 固件与驱动 | 1.0.17 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN      | 6.0.RC1 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | Python    | 3.7.5   | -          |
    | PyTorch   | 1.9.0 | -          |


----
# 快速上手

## 获取源码
1. 克隆开源仓源码
    ```shell
    git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection.git
    cd Ultra-Fast-Lane-Detection
    git checkout master
    git reset --hard f58fcd5f58511159ebfd06e60c7e221558075703
    touch model/__init__.py
    ```

2. 下载本仓，将本仓内所有文件放置于当前目录下，如果遇到同名文件覆盖即可

3. 执行以下命令创建 Python 虚拟环境并安装所需的依赖
    ```shell
    conda create -n UFLD python=3.7.5
    conda activate UFLD
    pip install -r requirements.txt
    ```

4. 创建一个目录，用于存放整个推理过程中所需文件与生成文件
    ```shell
    mkdir UFLD
    ```

## 准备数据集

1. 获取原始数据集  
    本推理项目使用 Tusimple 2782 张的测试集来验证模型精度，数据来自 [tusimple-benchmark](https://github.com/TuSimple/tusimple-benchmark/issues/3)。参考以下命令下载测试集、解压并整理：
    
    ```shell
    mkdir UFLD/TuSimple
    cd UFLD/TuSimple
    wget https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/test_set.zip
    wget https://s3.us-east-2.amazonaws.com/benchmark-frontend/truth/1/test_label.json
    unzip test_set.zip
    rm readme.md test_tasks_0627.json
    find clips -name '20.jpg' > test.txt
    cd ../../
    ```
    
    运行后，得到的数据集目录结构如下：
    ```
    ├── TuSimple/
        ├── clips/
            ├── 0530/
                ├── 1492626047222176976_0/
                    ├── 1.jpg
                    ├── ...
                    ├── 20.jpg
                ├── ...
            ├── 0531/
                ├── 1492626253262712112/
                    ├── 1.jpg
                    ├── ...
                    ├── 20.jpg
                ├── ...
            ├── 0601/
                ├── 1494452381594376146/
                    ├── 1.jpg
                    ├── ...
                    ├── 20.jpg
                ├── ...
        ├── test.txt
        ├── test_label.json
    ```
    
2. 数据预处理  
    执行前处理脚本将原始数据集中的 jpg 图片转换为 OM 模型输入需要的 bin 文件。
    ```shell
    python3 UFLD_preprocess.py --dataset-path UFLD/TuSimple/ --bin-path UFLD/TuSimple_bin
    ```
    参数说明：
    + --dataset-path: 原始测试集所在路径
    + --bin-path: 存放生成的bin文件的目录路径
    
    运行成功后，每张图像对应生成一个二进制 bin 文件，存放于 UFLD/TuSimple_bin 目录中。


## 模型转换

1. PyTroch 模型转 ONNX 模型  

    step1: 下载pth权重文件  
    本推理项目使用开源仓提供的预训练好的权重文件，可从[链接](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Ultra-Fast-Lane-Detection/PTH/tusimple_18.pth)获取，下载完成后将权重文件 tusimple_18.pth 存放于 UFLD 目录下。

    step2: 生成 ONNX 模型
    ```shell
    python3 UFLD_pth2onnx.py --model-path UFLD/tusimple_18.pth --onnx-path UFLD/tusimple_Dynamic.onnx
    ```
    参数说明：
    + --model-path: 预训练权重所在路径
    + --onnx-path: 生成ONNX模型的保存路径
    
    运行结束后，在 UFLD 目录下会生成 tusimple_Dynamic.onnx

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
        --model=UFLD/tusimple_Dynamic.onnx \
        --input_shape="input:${bs},3,288,800" \
        --output=UFLD/tusimple_bs${bs} \
        --output_type=FP16 \
        --insert_op_conf=./UFLD_aipp.config \
        --enable_small_channel=1 \
        --log=error \
        --soc_version=Ascend${chip_name}
    ```

   参数说明：
    + --model: 为ONNX模型文件。
    + --framework: 5代表ONNX模型。
    + --input_shape: 输入数据的shape。
    + --output: 输出的OM模型。
    + --output_type: 网络输出数据类型。
    + --log：日志级别。
    + --soc_version: 处理器型号。
    + --insert_op_conf: 插入算子的配置文件路径。

    命令中的`${bs}`表示模型输入的 batchsize，比如将`${bs}`设为 1024，则运行结束后会在 UFLD 目录下生成 tusimple_bs1024.om


## 推理验证

1. 安装ais_bench推理工具  

    请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。
    
2. 离线推理  

    使用ais_infer工具将预处理后的数据传入模型并执行推理：
    ```shell
    # 对预处理后的数据进行推理
    python3 -m ais_bench --model ./UFLD/tusimple_bs1.om --input ./UFLD/TuSimple_bin --output ./UFLD/result --output_dirname result_bs1
    ```
    参数说明：
    + --model: OM模型路径。
    + --input: 存放预处理bin文件的目录路径
    + --output: 存放推理结果的目录路径
    + --output_dirname: 推理结果输出子文件夹
    
    运行成功后，在 UFLD/result 下，会生成指定名字的的子目录，这里为result_bs1，用于存放推理结果文件。
    
3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的准确率：
    ```shell
    python3 UFLD_postprocess.py \
        --result-path UFLD/result/result_bs1/ \
        --label-path UFLD/TuSimple/test_label.json
    ```
    参数说明：
    +  --result-path: 生成推理结果所在路径。
    +  --label-path: 图片标签文件路径。
    
    运行成功后，程序会打印出模型的精度指标：
    ```
    Accuracy 0.9581343500735975
    FP 0.19033069734004313
    FN 0.039030673376467734
    ```

4. 性能验证  

    对于性能的测试，需要注意以下三点：
    + 测试前，请通过 npu-smi info  命令查看 NPU 设备状态，请务必在 NPU 设备空闲的状态下进行性能测试。
    + 为避免因测试持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps.

    > 吞吐率（throughput）：模型在单位时间（1秒）内处理的数据样本数。
    
    执行纯推理：
    ```shell
    python3 -m ais_bench --model ./UFLD/tusimple_bs${bs}.om --batchsize ${bs} --loop 100
    ```
    
    程序会打印出跟性能相关的指标。

----
# 精度&性能

1. 精度对比

    自测了 batchsize 为 1 和 1024 的精度，两个 batchsize 得到的精度没有差别，且与开源仓精度的相对误差小于 1%.
    <table>
    <tr>
    <th>Model</th>
    <th>batchsize</th>
    <th>Accuracy</th>
    <th>开源仓精度</th>
    <th>误差</th>
    </tr>
    <tr>
    <td rowspan="2">Ultra-Fast-Lane-Detection</td>
    <td>1</td>
    <td rowspan="2">95.81%</td>
    <td rowspan="2"><a href="https://github.com/cfzd/Ultra-Fast-Lane-Detection#trained-models">95.82%</a></td>
    <td rowspan="2"> $$ \frac {|0.9581-0.9582|} {0.9582}= 0.0001$$ </td>
    </tr>
    <tr>
    <td>1024</td>
    </tr>
    </table>  

2. 性能对比
  
    | batchsize | 数据集 | 310P性能（FPS） |
    | ---- | ---- | ---- |
    | 1 | Tusimple | 840.35 |
    | 4 | Tusimple | 1627.39 |
    | 8 | Tusimple | 1904.80 |
    | 16 | Tusimple | 2056.05 |
    | 32 | Tusimple | 2254.11 |
    | 64 | Tusimple | 1990.84 |
    | 128 | Tusimple | 2048.26 |
    | 256 | Tusimple | 2099.54 |
    | 512 | Tusimple | 2094.81         |
    | 1024 | Tusimple | 2103.10 |
