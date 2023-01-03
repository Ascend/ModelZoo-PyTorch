# SATRN 模型推理指导

- [SATRN 模型推理指导](#satrn-模型推理指导)
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

在我们的日常生活中存在非常多的严重弯曲或旋转的文本，场景文本识别（STR）在这些场景下已有很大的进步，但仍然不能识别任意形状的文本。受到 Transformer 自注意力机制的启发，作者提出了一种可以识别任意形状文本的新网络结构 SATRN。SATRN 通过自注意机制来描述文本图像中字符的二维空间依赖性。利用自注意机制的全图传播，SATRN 可以识别具有任意排列和大字符间距的文本。基于以上的优势，SATRN 在不规则文本图像上的表现比现有的 STR 模型平均高出 5.7pp.

+ 论文  
    [On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention](https://arxiv.org/abs/1910.04396)  
    Junyeop Lee, Sungrae Park, Jeonghun Baek, Seong Joon Oh, Seonghyeon Kim, Hwalsuk Lee

+ 参考实现  
    ```
    url = https://github.com/open-mmlab/mmocr/tree/main/configs/textrecog/satrn
    tag = v0.6.1
    commit_id = e5f071afb80d899c6c44eb95ac8e0357b492b369
    model_name = Satrn_small
    ```

+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | input      |  RGB_FP32 | NCHW        | 1 x 3 x 32 x 100 |

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output      |  FLOAT32   | ND          | 1 x 25 x 92 |



----
# 推理环境

- 该模型离线推理使用 Atlas 300I Pro 推理卡，推理所需配套的软件如下：

    | 配套      | 版本    | 环境准备指导 |
    | --------- | ------- | ---------- |
    | firmware  | 1.82.22.2.220 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | driver    | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN      | 5.1.RC2 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | Python    | 3.8   | -          |


----
# 快速上手

## 获取源码

1. 克隆开源仓源码
    ```shell
    git clone -b v0.6.1 https://github.com/open-mmlab/mmocr.git
    cd mmocr
    git reset --hard e5f071afb80d899c6c44eb95ac8e0357b492b369
    ```

2. 下载本仓，将该模型目录下的Python脚本、requirements.txt与补丁文件复制到当前目录，并通过补丁修改源码。
    ```shell
    patch -p1 < satrn.patch
    ```

3. 执行以下命令创建 Python 虚拟环境并安装所需的依赖
    ```shell
    pip3 install torch==1.10.0 torchvision
    pip3 install openmim
    mim install mmdet==2.25.1
    mim install mmcv==1.6.0
    pip3 install onnx-simplifier==0.4.0
    pip3 install tqdm 
    pip3 install decorator 
    pip3 install sympy
    pip3 install -e .
    ```

4. 创建一个目录，用于存放整个推理过程中所需文件与生成文件
    ```shell
    mkdir satrn
    ```

## 准备数据集

1. 获取原始数据集  
    该模型用 [IIIT5K](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html) 数据集的 3000 张测试图片来验证模型精度，请自行下载 [数据集](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz) 与 [label](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/test_label.txt) 文件，参考以下命令下载数据集并解压。
    ```shell
    mkdir -p data/mixture/
    cd data/mixture/
    wget http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz
    tar -zxvf IIIT5K-Word_V3.0.tar.gz
    wget https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/test_label.txt -P IIIT5K/
    cd ../../
    ```
    本推理项目需要的数据目录结构如下：
    ```
    data/mixture/IIIT5K/
    ├── test/
        ├── 1002_1.png
        ├── 1002_2.png
        ├── 1009_1.png
        ├── ...
    ├── test_label.txt
    ```

2. 数据预处理  
    执行前处理脚本将原始数据转换为 OM 模型输入需要的 bin 文件。
    ```shell
    python satrn_preprocess.py --cfg-path configs/textrecog/satrn/satrn_small.py --save-dir satrn/test_bin
    ```
    参数说明：
    + --cfg-path: 模型配置文件路径
    + --save-dir: 存放生成的bin文件的目录路径
    
    预处理脚本运行结束后，每张原始图片都会对应生成一个 bin 文件，共计生成 3000 个 bin 文件，存放于 satrn/test_bin 目录下。


## 模型转换

1. PyTroch 模型转 ONNX 模型  

    step1: 下载预训练权重
    下载 mmocr 官方提供的 [预训练权重](https://download.openmmlab.com/mmocr/textrecog/satrn/satrn_small_20211009-2cf13355.pth) 到 satrn 目录下
    ```shell
    wget https://download.openmmlab.com/mmocr/textrecog/satrn/satrn_small_20211009-2cf13355.pth -P satrn
    ```

    step2: 生成 ONNX 模型
    ```
    python satrn_pth2onnx.py \
        --cfg-path configs/textrecog/satrn/satrn_small.py \
        --ckpt-path satrn/satrn_small_20211009-2cf13355.pth \
        --onnx-path satrn/satrn.onnx
    ```
    参数说明：
    + --cfg-path: 模型配置文件路径
    + --ckpt-path: 模型预训练权重路径
    + --onnx-path: ONNX模型保存目录
    + --opset-version: ONNX算子集版本，默认值为11

    step3: 简化 ONNX 模型
    ```shell
    onnxsim satrn/satrn.onnx satrn/satrn_sim.onnx
    ```

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
        --model=satrn/satrn_sim.onnx \
        --input_shape="input:1,3,32,100" \
        --output=satrn/satrn_sim \
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

    因模型体量太大，执行上述命令后，需要等待至少半小时以上的时间才能转出OM模型。


## 推理验证

1. 准备推理工具  

    本推理项目使用 ais_bench 作为推理工具，须自己打包并安装。
    请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

2. 离线推理  

    使用ais_bench推理工具将预处理后的数据传入模型并执行推理：
    ```shell
    # 设置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /etc/profile
    
    # 对预处理后的数据进行推理
    mkdir satrn/test_out/
    python3 -m ais_bench --model satrn/satrn_sim.om --input satrn/test_bin --output satrn/test_out/ --batchsize 1
    ```
    参数说明：
    + --model: OM模型路径。
    + --input: 存放预处理bin文件的目录路径
    + --output: 存放推理结果的目录路径
    + --batchsize: 模型每次处理样本的数量
    
    运行成功后，在 satrn/test_out/ 下，会生成一个以执行开始时间`%Y_%m_%d-%H_%M_%S`来命名的子目录，每个预处理 bin 文件会对应生成一个推理结果 bin 文件存放在此目录下。
  
3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的准确率：
    ```shell
    python satrn_postprocess.py --result-dir satrn/test_out/2022_09_28-08_47_23/ --gt-path data/mixture/IIIT5K/test_label.txt
    ```
    参数说明：
    + --result-dir: 存放推理结果的目录路径
    + --gt-path: groundtruth文件路径
    
    运行成功后，程序会打印出模型的精度指标：
    ```
     {'1-N.E.D': 0.9826,
      'char_precision': 0.9812,
      'char_recall': 0.9828,
      'word_acc': 0.0647,
      'word_acc_ignore_case': 0.845,
      'word_acc_ignore_case_symbol': 0.9487}
    ```

4. 性能验证  

    对于性能的测试，需要注意以下三点：
    + 测试前，请通过 npu-smi info  命令查看 NPU 设备状态，请务必在 NPU 设备空闲的状态下进行性能测试。
    + 为避免因测试持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps.

    > 吞吐率（throughput）：模型在单位时间（1秒）内处理的数据样本数。
    
    执行纯推理：
    ```shell
    python3 -m ais_bench --model satrn/satrn_sim.om --loop 100 --batchsize 1
    ```

    执行完纯推理命令，程序会打印出与性能相关的指标：
    ```
    [INFO] -----------------Performance Summary------------------
    [INFO] H2D_latency (ms): min = 0.4661083221435547, max = 0.4661083221435547, mean = 0.4661083221435547, median = 0.4661083221435547, percentile(99%) = 0.4661083221435547
    [INFO] NPU_compute_time (ms): min = 31.332000732421875, max = 32.96500015258789, mean = 31.739540023803713, median = 31.685500144958496, percentile(99%) = 32.188839759826664
    [INFO] D2H_latency (ms): min = 0.2899169921875, max = 0.2899169921875, mean = 0.2899169921875, median = 0.2899169921875, percentile(99%) = 0.2899169921875
    [INFO] throughput 1000*batchsize(1)/NPU_compute_time.mean(31.739540023803713): 31.506442728849557
    [INFO] ------------------------------------------------------

    ```
    
    计算吞吐率：
    + 执行纯推理时若指定了 batchsize，则找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率，本例中的吞吐率为 31.506442728849557 fps.
    + 若没有指定 batchsize，则可以通过 **NPU_compute_time** 中的 **mean** 来计算：
    $$throughput =\frac{batchsize}{mean} * 1000 = 31.51 (fps)$$

----
# 性能&精度

1. 模型在 310P 设备上的性能是 T4 设备上性能的 0.677 倍，已通过性能评审。
2. 模型在 310P 设备上的精度优于开源仓提供的精度，高出 0.17 个百分点。

指标详情如下：

| 模型名 | batchsize | 310P实测精度 | 开源仓精度 |T4性能 | 310P性能 | 310P/T4 |
|-------| --------- | -------- | --------- | -------- | ---- | ---- |
|Satrn_small| 1 | 94.87% | [94.7%](https://github.com/open-mmlab/mmocr/tree/main/configs/textrecog/satrn#results-and-models) | 46.53 fps | 31.51 fps | 0.677倍 |

注：该推理模型只支持 batchsize 为 1.
