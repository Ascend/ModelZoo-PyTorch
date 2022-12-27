#  FOTS 模型推理指导

- [FOTS 模型推理指导](#fots-模型推理指导)
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

与以前的方法相比，FOTS 引入了新颖的 ROIRotate 操作，将文本检测和识别统一成端到端框架。TOTS 不仅适用于水平文本，并且可以解决更复杂和困难的情况。网络中共享训练特征，互补监督的应用，加快了模型整体的速度，使得 FOTS 在速度和性能方面完全碾压了 CTPN，可以进行实时的文本识别。

+ 论文  
    [FOTS: Fast Oriented Text Spotting with a Unified Network](https://arxiv.org/abs/1801.01671)  
    Xuebo Liu, Ding Liang, Shi Yan, Dagui Chen, Yu Qiao, Junjie Yan

+ 参考实现  
    ```
    url = https://github.com/Wovchena/text-detection-fots.pytorch.git
    branch = random-search
    commit_id = ef2d41e3bb911a4d032a34bec79dea5630627a8d
    ```

+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | image      |  RGB_FP32 | NCHW        | batchsize x 3 x 1248 x 2240 |

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | location    |  FLOAT32   | NCHW        | batchsize x 1 x 312 x 560 |
    | -           |  FLOAT32   | NCHW        | batchsize x 4 x 312 x 560 |
    | -           |  FLOAT32   | NCHW        | batchsize x 1 x 312 x 560 |

----
# 推理环境

- 该模型离线推理使用 Atlas 300I Pro 推理卡，推理所需配套的软件如下：

    | 配套      | 版本    | 环境准备指导 |
    | --------- | ------- | ---------- |
    | 固件与驱动  | 22.0.2 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN      | 5.1.RC2 | -          |
    | Python    | 3.7.13  | -          |

    说明：该模型离线推理使用 Atlas 300I Pro 推理卡，Atlas 300I Duo 推理卡请以 CANN 版本选择实际固件与驱动版本。


----
# 快速上手

## 获取源码


1. 克隆开源仓源码
    ```shell
    git clone https://github.com/Wovchena/text-detection-fots.pytorch.git
    cd text-detection-fots.pytorch/
    git checkout random-search
    git reset --hard ef2d41e3bb911a4d032a34bec79dea5630627a8d
    ```

2. 安装推理过程所需的依赖
    ```shell
    pip install -r requirements.txt
    ```

3. 下载本仓，将该模型目录下的Python脚本、requirements.txt与补丁文件复制到当前目录，并修改源码。
    ```shell
    patch -p1 < FOTS.patch
    ```


## 准备数据集

1. 获取原始数据集  
   
    该推理模型使用 **ICDAR2015** 测试集来验证模型精度。请按照以下的步骤获取原始测试数据：  
    step1: 进入 [下载页面](https://rrc.cvc.uab.es/?ch=4&com=downloads), 注册并登录  
    step2: 点击页面顶部 Chanllenges 下拉框，选择 Incidental Scene Text  
    step3: 顶部菜单栏点击 Downloads，进入下载页面  
    step4: 下载 Task 4.1 下的 **Test Set images** 和 **Test Set Ground Truth**  
    step5: 将 ch4_test_images.zip 与 Challenge4_Test_Task1_GT.zip 上传到当前目录  
    step6: 解压：`unzip ch4_test_images.zip -d ch4_test_images`
    
    执行完上述步骤后，在当前目录下的数据目录结构为：
    ```
    ./
    ├── Challenge4_Test_Task1_GT.zip
    ├── ch4_test_images/
        ├── img_1.jpg
        ├── img_10.jpg
        ├── ...
    ```

2. 数据预处理  
    执行前处理脚本将原始数据集中的 jpg 图片转换为 OM 模型输入需要的 bin 文件。
    ```shell
    mkdir res
    python FOTS_preprocess.py --images-folder ./ch4_test_images/ --output-folder ./res/
    ```
    参数说明：
    + --images-folder: 原始图片所在目录路径
    + --output-folder: 存放生成的bin文件的目录路径
    
    预处理脚本运行结束后，每张原始图都会对应生成一个 bin 文件，存放于 res 目录中。


## 模型转换

1. PyTroch 模型转 ONNX 模型  

    step1: 下载pth权重文件  
    本推理项目使用开源仓提供的预训练好的权重，进入[下载页面](https://drive.google.com/drive/folders/1xaVshLRrMEkb9LA46IJAZhlapQr3vyY2), 选择 epoch_582_checkpoint.pt 文件，下载后导入到当前目录。


    step2: 生成 ONNX 模型
    ```shell
    python ./FOTS_pth2onnx.py ./epoch_582_checkpoint.pt ./FOTS.onnx
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
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/Ascend/driver/lib64/driver
    
    # 执行 ATC 进行模型转换
    atc --framework=5 \
        --model=./FOTS.onnx \
        --output=./FOTS_bs${bs} \
        --input_format=NCHW \
        --input_shape="image:${bs},3,1248,2240" \
        --log=error \
        --soc_version=Ascend${chip_name} \
        --enable_small_channel=1 \
        --insert_op_conf=FOTS_aipp.cfg
    ```

   参数说明：
    + --model: 为ONNX模型文件。
    + --framework: 5代表ONNX模型。
    + --input_shape: 输入数据的shape。
    + --input_format: 输入数据的排布格式。
    + --output: 输出的OM模型。
    + --log：日志级别。
    + --soc_version: 处理器型号。

    命令中的`${bs}`表示模型输入的 batchsize，比如将`${bs}`设为 1，则运行结束后会在当前目录下生成 FOTS_bs1.om


## 推理验证

1. 准备推理工具  

    本推理项目使用ais_bench 作为推理工具。  
    请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。  

2. 离线推理  

    使用ais_bench工具将预处理后的数据传入模型并执行推理：
    ```shell
    # 设置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/Ascend/driver/lib64/driver
    
    # 对预处理后的数据进行推理
    mkdir result
    python -m ais_bench \
        --model ./FOTS_bs1.om \
        --input ./res \
        --output ./result \
        --batchsize 1 \
    ```
    参数说明：
    + --model: OM模型路径。
    + --input: 存放预处理bin文件的目录路径
    + --output: 存放推理结果的目录路径
    + --batchsize: 每次输入模型的样本数
    
    运行成功后，在 result 下，会生成一个以执行开始时间`%Y_%m_%d-%H_%M_%S`来命名的子目录，每个预处理 bin 文件会对应生成三个推理结果 bin 文件存放在此目录下。
  
3. 精度验证  

    首先调用 FOTS_postprocess.py 对模型的推理输出进行处理：
    ```shell
    mkdir outPost
    python FOTS_postprocess.py ./outPost/ ./result/2022_09_30-03_05_04/
    ```

    调用开源仓原 icdar_eval 目录下的 script.py 获取推理模型的精度指标：
    ```shell
    mkdir runs
    zip -jmq runs/u.zip outPost/*
    python icdar_eval/script.py -g=Challenge4_Test_Task1_GT.zip -s=runs/u.zip
    ```
    
    运行成功后，结果会直接打印出指标：
    ```
    Calculated!{"precision": 0.8642105263157894, "recall": 0.7905633124699085, "hmean": 0.8257480512946, "AP": 0}
    ```

4. 性能验证  

    对于性能的测试，需要注意以下三点：
    + 测试前，请通过 npu-smi info  命令查看 NPU 设备状态，请务必在 NPU 设备空闲的状态下进行性能测试。
    + 为避免因测试持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps.

    > 吞吐率（throughput）：模型在单位时间（1秒）内处理的数据样本数。
    
    执行纯推理：
    ```shell
    python -m ais_bench --model ./FOTS_bs1.om --loop 100 --batchsize 1
    ```

    执行完纯推理命令，程序会打印出与性能相关的指标。找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

----
# 性能&精度

1. 性能对比

| Batch Size | 310 (FPS/Card) | 310P (FPS/Card) | T4 (FPS/Card) | 310P/310   | 310P/T4   |
| ---------- | -------------- | ------------- | -------- | -------- | -------- |
| 1          | *35.70*        | *85.73*       | *44.89*  | *2.40* | *1.91* |
| 2          | *31.62*        | *55.82*       | *46.08*  | *1.77* | *1.21* |
| 4          | *31.37*        | *56.45*       | *46.91*  | *1.79* | *1.20* |
| 8          | *31.22*        | *53.09*       | *46.36*  | *1.70* | *1.15* |
| **best**   | **35.70**      | **85.73**     | **46.91**| **2.40** | **1.83** |

- 在310P设备上，当 batchsize为1 时模型性能最优，达 85.73 fps
- 最优性能相比，310P/310=2.40倍，310P/T4=1.83倍
- 因显存不足，不支持推理 batchsize >= 16 的模型

2. 精度对比

| Metrics   | OM精度      | 开源仓精度   |
| ----------| -----------| -------------|
| precision | 0.864      | 0.869        |
| recall    | 0.791      | 0.799        |
| hmean     | 0.826      | 0.833        |
| AP        | 0          | 0            |

