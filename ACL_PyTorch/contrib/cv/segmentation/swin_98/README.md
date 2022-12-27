#  swin_98 模型推理指导

- [swin\_98 模型推理指导](#swin_98-模型推理指导)
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

Transformer 在 NLP 领域表现优异，如何将 Transformer 从 NLP 领域应用到 CV 领域？其挑战来自两个领域在尺度与分辨率上的差异。NLP 任务中每个词向量的维度是固定的，而 CV 任务中往往图像尺度变化较大；且与文本段落中的单词量相比，图像中的像素分辨率要高得多。为了解决这些问题，作者提出了一种分层 Transformer，通过 Shifted windows(移位窗口) 将自注意力的计算限制在不重叠的局部窗口范围内，同时允许跨窗口连接，从而带来更高的效率。这种分层架构具有在各种尺度上建模的灵活性，且只有相对于图像大小的线性计算复杂度。Swin Transformer 的这些特性使其与广泛的 CV 任务兼容，包括图像分类和密集预测任务，例如目标检测和语义分割。在这些任务上的优异表现说明，Swin Transformer 可以作为 CV 领域的通用主干网络。


+ 论文  
    [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)  
    [Ze Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z), [Yutong Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+Y), [Yue Cao](https://arxiv.org/search/cs?searchtype=author&query=Cao%2C+Y), [Han Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+H), [Yixuan Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+Y), [Zheng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Stephen Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+S), [Baining Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+B)

+ 参考实现  
    ```
    url = https://github.com/open-mmlab/mmsegmentation/tree/master/configs/swin
    branch = v0.28.0
    commit_id = b51670b61339e5b10c5ab6c277de6b6a387fdff0
    model_name = upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K
    ```

+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | input      |  RGB_FP32 | NCHW        | batchsize x 3 x 512 x 512 |

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output      |  FLOAT32   | ND          | batchsize x 150 x 512 x 512 |



----
# 推理环境

- 该模型离线推理使用 Atlas 300I Pro 推理卡，推理所需配套的软件如下：

    | 配套      | 版本    | 环境准备指导 |
    | --------- | ------- | ---------- |
    | firmware  | 1.82.22.2.220 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | driver    | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN      | 5.1.RC2 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | Python    | 3.8     | -          |
    | PyTorch   | 1.10.0  | -          |


----
# 快速上手

## 获取源码


1. 克隆开源仓源码
    ```shell
    git clone -b v0.28.0 https://github.com/open-mmlab/mmsegmentation.git
    cd mmsegmentation
    git reset --hard b51670b61339e5b10c5ab6c277de6b6a387fdff0
    ```

2. 执行以下命令创建 Python 虚拟环境并安装所需的依赖
    ```shell
    conda create -n swin98 python=3.8
    conda activate swin98
    pip3 install torch==1.10.0 torchvision
    pip3 install openmim
    mim install mmcv-full==1.6.0
    pip3 install tqdm
    pip3 install decorator
    pip3 install sympy
    pip3 install -v -e .
    ```

3. 下载本仓，将本仓内的 Python 脚本放置于当前目录下

4. 创建一个目录，用于存放整个推理过程中所需文件与生成文件
    ```shell
    mkdir swin98
    ```

## 准备数据集

1. 获取原始数据集  
    本推理项目使用 ADE20K 的 2000 张验证集图片来验证模型精度，请进入 [ADE20K官网](http://groups.csail.mit.edu/vision/datasets/ADE20K/) 自行下载数据集（需要先注册）。下载后请自行解压或参考以下命令：
    ```shell
    mkdir -p data/ade
    unzip /path/to/ADEChallengeData2016.zip -d data/ade/
    ```
    最终，验证集原始图片与标注图片的存放结构如下：
    ```
    ├── data/ade/ADEChallengeData2016/
        ├── annotations/
            ├── validation/
                ├── ADE_val_00000001.png
                ├── ...
                ├── ADE_val_00002000.png
        ├── images/
            ├── validation/
                ├── ADE_val_00000001.jpg
                ├── ...
                ├── ADE_val_00002000.jpg
    ```

2. 数据预处理  
    执行前处理脚本将原始数据集中的 jpg 图片转换为 OM 模型输入需要的 bin 文件。
    ```shell
    python swin98_preprocess.py \
        --config configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py \
        --save-dir swin98/val_bin/
    ```
    参数说明：
    + --config: 模型配置文件路径
    + --save-dir: 存放生成的bin文件的目录路径
    
    原始图片在预处理的时候会进行滑窗操作，一张图片对应一个或多个滑窗，每个滑窗单独保存成一个 bin 文件。预处理脚本运行结束后，2000 张原始图会生成 3686 个 bin 文件，存放于 swin98/val_bin 目录中。


## 模型转换

1. PyTroch 模型转 ONNX 模型  

    step1: 下载pth权重文件  
    本推理项目使用开源仓提供的预训练好的 [权重文件](https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192015-ee2fff1c.pth)，下载完成后将权重 pth 文件存放于 swin98 目录下。
    ```shell
    wget https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192015-ee2fff1c.pth -P swin98
    ```

    step2: 生成 ONNX 模型
    ```shell
    python swin98_pth2onnx.py \
        --config configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py \
        --checkpoint swin98/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192015-ee2fff1c.pth \
        --onnx swin98/small_slide_bs${bs}.onnx \
        --batchsize ${bs}
    ```
    参数说明：
    + --config: 模型配置文件路径
    + --checkpoint: 预训练权重所在路径
    + --onnx: 生成ONNX模型的保存路径
    + --batchsize: 模型输入的batchsize，默认为 1
    + --opset-version: ONNX算子集版本，默认为 11
    
    命令中的`${bs}`表示模型输入的 batchsize，比如将`${bs}`设为 1，运行结束后，在 swin98 目录下会生成 small_slide_bs1.onnx

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
        --model=swin98/small_slide_bs${bs}.onnx \
        --input_shape="input:${bs},3,512,512" \
        --output=swin98/small_slide_bs${bs} \
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

    命令中的`${bs}`表示模型输入的 batchsize，比如将`${bs}`设为 1，则运行结束后会在 swin98 目录下生成 small_slide_bs1.om


## 推理验证

1. 安装ais_bench推理工具  

    请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

2. 离线推理  

    使用ais_bench推理工具将预处理后的数据传入模型并执行推理：
    ```shell
    # 设置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /etc/profile
    
    # 对预处理后的数据进行推理
    mkdir swin98/val_bs1_out/
    cd ais_infer
    python3 -m ais_bench --model ../swin98/small_slide_bs1.om --input ../swin98/val_bin/ --output ../swin98/val_bs1_out/ --batchsize 1
    cd ..
    ```
    参数说明：
    + --model: OM模型路径。
    + --input: 存放预处理bin文件的目录路径
    + --output: 存放推理结果的目录路径
    + --batchsize: 模型一次处理样本的数量
    
    运行成功后，在 swin98/val_bs1_out/ 下，会生成一个以执行开始时间`%Y_%m_%d-%H_%M_%S`来命名的子目录，每个预处理 bin 文件会对应生成一个推理结果 bin 文件存放在此目录下。
  
3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的准确率：
    ```shell
    python swin98_postprocess.py \
        --config configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py \
        --infer-results swin98/val_bs1_out/2022_09_24-07_26_31/
    ```
    参数说明：
    +  --config: 模型配置文件路径
    +  --infer-results: 存放推理结果的目录路径
    
    运行成功后，程序会打印出模型在每个类别（共150类）上的精度指标以及整体的精度指标：
    ```
    per class results:
    +---------------------+-------+-------+
    |        Class        |  IoU  |  Acc  |
    +---------------------+-------+-------+
    |         wall        | 76.88 | 88.54 |
    |       building      | 81.81 | 91.75 |
    |         ...         |  ...  |  ...  |
    |        clock        | 38.05 | 47.48 |
    |         flag        | 44.42 | 49.01 |
    +---------------------+-------+-------+
    
    Summary:
    +-------+-------+-------+
    |  aAcc |  mIoU |  mAcc |
    +-------+-------+-------+
    | 82.49 | 47.92 | 59.5 |
    +-------+-------+-------+
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
    python3 -m ais_bench --model ../swin98/small_slide_bs1.om --loop 100 --batchsize 1
    cd ..
    ```

    执行完纯推理命令，程序会打印出与性能相关的指标：
    ```
    [INFO] -----------------Performance Summary------------------
    [INFO] H2D_latency (ms): min = 0.7932186126708984, max = 0.7932186126708984, mean = 0.7932186126708984, median = 0.7932186126708984, percentile(99%) = 0.7932186126708984
    [INFO] NPU_compute_time (ms): min = 59.518001556396484, max = 60.215999603271484, mean = 59.75496997833252, median = 59.730499267578125, percentile(99%) = 60.15362987518311
    [INFO] D2H_latency (ms): min = 98.02532196044922, max = 98.02532196044922, mean = 98.02532196044922, median = 98.02532196044922, percentile(99%) = 98.02532196044922
    [INFO] throughput 1000*batchsize(1)/NPU_compute_time.mean(59.75496997833252): 16.73500966300553
    [INFO] ------------------------------------------------------

    ```
    
    计算吞吐率：
    + 执行纯推理时若指定了 batchsize，则找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率，本例中的吞吐率为 16.73500966300553
    + 若没有指定 batchsize，则可以通过 **NPU_compute_time** 中的 **mean** 来计算：
    $$throughput =\frac{batchsize}{mean} * 1000 = 16.74(fps)$$

----
# 性能&精度

1. 性能对比
  
    在 310P 设备上，当 batchsize 为 1 时模型性能最优，达 16.74 fps.
    | batchsize | T4性能 | 310P性能 | 310P/T4 |
    | ---- | ---- | ---- | ---- |
    | 1 | 5.24 fps | 16.74 fps | 3.19倍 |
    | 4 | 4.65 fps | 16.06 fps | 3.45倍 |
    | 8 | 4.35 fps | 16.20 fps | 3.72倍 | 
    | **best** | **5.24 fps** | **16.74 fps** | **3.19倍** |
    
    注：当 batchsize 为 16 或更高时，因内存不足导致推理失败，无法获取性能数据。


2. 精度对比

    自测了 batchsize 为 1 和 4 的精度，两个 batchsize 得到的精度没有差别，且比开源仓精度的高出 0.20%.
    <table>
    <tr>
    <th>Model</th>
    <th>batchsize</th>
    <th>mIoU(NPU)</th>
    <th>mIoU(开源仓)</th>
    </tr>
    <tr>
    <td rowspan="2">swin_98</td>
    <td>1</td>
    <td rowspan="2">47.92%</td>
    <td rowspan="2"><a href="https://github.com/open-mmlab/mmsegmentation/tree/v0.28.0/configs/swin#ade20k">47.72%</a></td>
    </tr>
    <tr>
    <td>4</td>
    </tr>
    </table>  
