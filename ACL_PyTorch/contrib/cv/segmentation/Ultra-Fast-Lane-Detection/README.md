
# Ultra-Fast-Lane-Detection 模型推理指导

- [概述](#概述)
- [推理环境](#推理环境)
- [快速上手](#快速上手)
    - [获取源码](#获取源码)
    - [准备数据集](#准备数据集)
    - [模型转换](#模型转换)
    - [推理验证](#推理验证)
- [精度&性能](#精度性能)

----
# 概述

现行主流的车道检测方法将车道检测视为像素分割问题，难以解决高挑战性的场景与速度问题。受人类感知的启发，严重遮挡和极端光照条件下的车道识别主要基于上下文和全局信息。由此，作者针对极快的速度和复杂的场景提出了一种新颖、简单但有效的方法。具体来说，作者将车道检测过程视为使用全局特征的 **row-based selecting**，这使得计算成本显著降低。使用全局特征的大感受野，也可以处理高挑战性的场景。此外，作者还提出了 **structural loss** 用以显式地模拟车道结构。在两个车道检测基准数据集上进行的大量实验表明，此方法在速度和准确性方面都可以达到目前的最佳水平。轻量级版本甚至可以在相同分辨率下达到每秒 300+ 帧，这比以前最先进的方法至少快 4 倍。

+ 论文  
    [Ultra Fast Structure-aware Deep Lane Detection](https://arxiv.org/abs/2004.11757)  
    [Zequn Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+Z), [Huanyu Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+H), [Xi Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+X)

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
    | output      |  FLOAT32   | ND          | batchsize x 22624 |



----
# 推理环境

- 该模型离线推理使用 Atlas 300I Pro 推理卡，推理所需配套的软件如下：

    | 配套      | 版本    | 环境准备指导 |
    | --------- | ------- | ---------- |
    | firmware  | 1.82.22.2.220 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | driver    | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN      | 5.1.RC2 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | Python    | 3.7.5   | -          |
    | PyTorch   | 1.9.1  | -          |


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
    python UFLD_preprocess.py --dataset-path UFLD/TuSimple/ --bin-path UFLD/TuSimple_bin
    ```
    参数说明：
    + --dataset-path: 原始测试集所在路径
    + --bin-path: 存放生成的bin文件的目录路径
    
    运行成功后，每张图像对应生成一个二进制 bin 文件，存放于 UFLD/TuSimple_bin 目录中。


## 模型转换

1. PyTroch 模型转 ONNX 模型  

    step1: 下载pth权重文件  
    本推理项目使用开源仓提供的预训练好的权重文件，可从 [GoogleDrive](https://drive.google.com/file/d/1WCYyur5ZaWczH15ecmeDowrW30xcLrCn/view?usp=sharing)/[BaiduDrive(code:bghd)](https://pan.baidu.com/s/1Fjm5yVq1JDpGjh4bdgdDLA) 获取，下载完成后将权重文件 tusimple_18.pth 存放于 UFLD 目录下。

    step2: 生成 ONNX 模型
    ```shell
    python UFLD_pth2onnx.py --model-path UFLD/tusimple_18.pth --onnx-path UFLD/tusimple_Dynamic.onnx
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

1. 准备推理工具  

    本推理项目使用 [ais_infer](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer#%E4%BB%8B%E7%BB%8D) 作为推理工具，须自己拉取源码，打包并安装。
    ```shell
    # 指定CANN包的安装路径
    export CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest

    # 获取推理工具源码
    git clone https://gitee.com/ascend/tools.git
    # cd tools/ais-bench_workload/tool/ais_infer/backend/
    cp -r tools/ais-bench_workload/tool/ais_infer .

    # 打包
    cd ais_infer/backend/
    pip3 wheel ./   # 会在当前目录下生成 aclruntime-xxx.whl，具体文件名因平台架构而异
    
    # 安装
    pip3 install --force-reinstall aclruntime-xxx.whl
    cd ../..
    ```

2. 离线推理  

    使用ais_infer工具将预处理后的数据传入模型并执行推理：
    ```shell
    # 设置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /etc/profile
    
    # 对预处理后的数据进行推理
    mkdir UFLD/bs1024_infer_result
    cd ais_infer
    python3 ais_infer.py --model ../UFLD/tusimple_bs1024.om --input ../UFLD/TuSimple_bin --output ../UFLD/bs1024_infer_result
    cd ..
    ```
    参数说明：
    + --model: OM模型路径。
    + --input: 存放预处理bin文件的目录路径
    + --output: 存放推理结果的目录路径
    
    运行成功后，在 UFLD/bs1024_infer_result 下，会生成一个以执行开始时间来命名的子目录，用于存放推理结果文件。
  
3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的准确率：
    ```shell
    python UFLD_postprocess.py \
        --result-path UFLD/bs1024_infer_result/2022_09_12-06_28_06/ \
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
    cd ais_infer
    python3 ais_infer.py --model ../UFLD/tusimple_bs1024.om --batchsize 1024 --loop 100
    cd ..
    ```

    执行完纯推理命令，程序会打印出跟性能先关的指标：
    ```
    [INFO] -----------------Performance Summary------------------
    [INFO] H2D_latency (ms): min = 165.70281982421875, max = 165.70281982421875, mean = 165.70281982421875, median = 165.70281982421875, percentile(99%) = 165.70281982421875
    [INFO] NPU_compute_time (ms): min = 482.4289855957031, max = 520.8070068359375, mean = 511.8491110229492, median = 512.0014953613281, percentile(99%) = 516.9628411865235
    [INFO] D2H_latency (ms): min = 27.114152908325195, max = 27.114152908325195, mean = 27.114152908325195, median = 27.114152908325195, percentile(99%) = 27.114152908325195
    [INFO] throughput 1000*batchsize(1024)/NPU_compute_time.mean(511.8491110229492): 2000.589583819924
    [INFO] ------------------------------------------------------

    ```
    
    计算吞吐率：
    + 执行纯推理时若指定了 batchsize，则找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率，本例中的吞吐率为 2000.589583819924
    + 若没有指定 batchsize，则可以通过 **NPU_compute_time** 中的 **mean** 来计算：
    $$throughput =\frac{batchsize}{mean} * 1000 = 2000.59(fps)$$

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
    <td rowspan="2">SE-ResNetXt101</td>
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
  
    在 310P 设备上，当 batchsize 为 1024 时模型性能最优，达 2000.59 fps.
    | batchsize | 310性能 | T4性能 | 310P性能 | 310P/310 | 310P/T4 |
    | ---- | ---- | ---- | ---- | ---- | ---- |
    | 1 | 395.26 fps | 641.03 fps | 826.97 fps | 2.09倍 | 1.29倍 |
    | 4 | 804.99 fps | 987.65 fps | 1556.10 fps | 1.93倍 | 1.58倍 |
    | 8 | 930.23 fps | 1126.76 fps | 1818.50 fps | 1.95倍 | 1.61倍 |
    | 16 | 1013.62 fps | 1209.4 fps | 1805.70 fps | 1.78倍 | 1.49倍 |
    | 32 | 1098.15 fps | 1217.66 fps | 1935.87 fps | 1.76倍 | 1.59倍 |
    | 64 | 1108.03 fps | 1259.35 fps | 1922.47 fps | 1.74倍 | 1.53倍 |
    | 128 | 1111.91 fps | 1278.98 fps | 1933.27 fps | 1.74倍 | 1.51倍 |
    | 256 | 1112.81 fps | 1286.43 fps | 1971.58 fps | 1.77倍 | 1.53倍 |
    | 512 | 1114.04 fps | 1320.44 fps | 1973.97 fps | 1.77倍 | 1.49倍 |
    | 1024 | - | - | 2000.59 fps | - | - |
    | **性能最优bs** | **1114.04 fps** | **1320.44 fps** | **2000.59 fps** | **1.80倍** | **1.52倍** |
    
    注：当batchsize为1024时，在310与T4设备因内存不足导致推理失败，无法获取性能数据。
