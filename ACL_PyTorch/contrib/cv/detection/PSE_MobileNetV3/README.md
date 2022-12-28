# PSE_MobileNetV3 模型推理指导

- [概述](#概述)
- [推理环境](#推理环境)
- [快速上手](#快速上手)
    - [获取源码](#获取源码)
    - [准备数据集](#准备数据集)
    - [模型转换](#模型转换)
    - [推理验证](#推理验证)
- [性能&精度](#性能精度)

----
# 概述

与传统的文本检测算法相比，PSENet具有以下两个优点。首先，作为一种基于分割的方法，PSENet执行像素级分割，该分割能够精确定位具有任意形状的文本实例。其次，PSENet是一种渐进式尺度扩展算法，利用该算法可以成功识别相邻的文本实例。


+ 论文  
    [Shape Robust Text Detection with Progressive Scale Expansion Network](https://arxiv.org/abs/1903.12473)  
    Wenhai Wang, Enze Xie, Xiang Li, Wenbo Hou, Tong Lu, Gang Yu, Shuai Shao


+ 参考实现  
    ```
    url = https://github.com/PaddlePaddle/PaddleOCR.git
    branch = release/2.6
    commit_id = 76fefd56f86f2809a6e8f719220746442fbab9f3
    model_name = PSE_MobileNetV3
    ```

+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | x  |  RGB_FP32 | NCHW | batchsize x 3 x 736 x 1312 |

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | maps |  FLOAT32   | ND   | batchsize x 7 x 184 x 328 |



----
# 推理环境

- 该模型推理所需配套的软件如下：

    | 配套      | 版本    | 环境准备指导 |
    | --------- | ------- | ---------- |
    | 固件与驱动 | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN      | 6.0.RC1 | -          |
    | Python    | 3.7.5   | -          |
    
    说明：请根据 CANN 版本选择相匹配的固件与驱动版本。


----
# 快速上手

## 获取源码

1. 克隆开源仓源码
    ```shell
    git clone -b release/2.6 https://github.com/PaddlePaddle/PaddleOCR.git
    cd PaddleOCR
    git reset --hard 76fefd56f86f2809a6e8f719220746442fbab9f3
    ```

2. 下载本仓，将该模型目录下的所有文件复制到当前目录，并修改源码。
    ```shell
    patch -p1 < pse.patch
    ```

3. 执行以下命令创建 Python 虚拟环境并安装所需的依赖
    ```shell
    pip3 install -r requirements.txt
    python3 setup.py install
    ```

4. 创建一个目录，用于存放整个推理过程中所需文件与生成文件
    ```shell
    mkdir pse
    ```

## 准备数据集

1. 获取原始数据集  
    该推理模型使用 ICDAR2015 测试集的500张图片来验证模型精度。请按照以下的步骤准备原始测试数据：  
    step1: 进入[下载页面](https://gitee.com/link?target=https%3A%2F%2Frrc.cvc.uab.es%2F%3Fch%3D4%26com%3Ddownloads), 注册并登录  
    step2: 点击页面顶部 Chanllenges 下拉框，选择 Incidental Scene Text  
    step3: 点击顶部菜单栏 Downloads 按钮，进入下载页面  
    step4: 下载 Task 4.1 下的 Test Set images, 上传到 pse/icdar2015 目录下  
    step5: 解压：`unzip pse/icdar2015/ch4_test_images.zip -d pse/icdar2015/ch4_test_images`  
    step6: 下载 PaddleOCR 标注：`wget https://paddleocr.bj.bcebos.com/dataset/test_icdar2015_label.txt -P pse/icdar2015`

    执行完上述步骤后，在当前目录下的数据目录结构为：   
    ```
    pse/icdar2015/
    ├── ch4_test_images/
        ├── img_1.jpg
        ├── img_10.jpg
        ├── ...
    ├── test_icdar2015_label.txt
    ```


2. 数据预处理  
    执行前处理脚本将原始数据转换为 OM 模型输入需要的 bin 文件。

    ```shell
    python3 pse_preprocess.py \
        --config configs/det/det_mv3_pse.yml \
        --opt data_dir=pse/icdar2015/ bin_dir=pse/bin_list info_dir=pse/info_list
    ```
    参数说明：
    + -c, --config: 模型配置文件路径
    + --opt data_dir: 原始数据集所在目录路径
    + --opt bin_dir: 存放生成的bin文件的目录路径
    + --opt info_dir: 存放groundtruth等信息的目录路径
    
    运行成功后，pse/bin_list 目录下会生成 500 个 bin 文件。 pse/info_list 目录下也会对应生成 500 个 pickle 文件


## 模型转换

1. PyTorch 模型转 ONNX 模型  

    step1: 下载 paddle 预训练模型
    下载 PaddleOCR 提供的 [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_mv3_pse_v2.0_train.tar) 到 pse 目录下，然后解压。
    ```shell
    cd pse
    wget https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_mv3_pse_v2.0_train.tar
    tar xf det_mv3_pse_v2.0_train.tar
    cd ..
    ```

    step2: paddle 训练模型转 paddle 推理模型
    ```
    python3 tools/export_model.py \
        -c configs/det/det_mv3_pse.yml \
        -o Global.pretrained_model=pse/det_mv3_pse_v2.0_train/best_accuracy \
           Global.save_inference_dir=pse/det_mv3_pse_v2.0_infer/
    ```
    参数说明：
    + -c, --config: paddle模型配置文件路径
    + -o Global.pretrained_model: paddle预训练模型路径
    + -o Global.save_inference_dir: paddle推理模型保存目录

    step3: 生成 ONNX 模型
    ```shell
    paddle2onnx \
        --model_dir pse/det_mv3_pse_v2.0_infer/ \
        --model_filename inference.pdmodel \
        --params_filename inference.pdiparams \
        --save_file pse/PSE_MobileNetV3.onnx \
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

    step4: 修改ONNX

    安装auto_optimizer

    ```shell
    git clone https://gitee.com/ascend/auto-optimizer.git
    cd auto-optimizer
    git reset --hard 52e2f791ba6fa9d55c5b57f0a2521c69aed01ea9
    pip3 install -r requirements.txt
    python3 setup.py install

    ```
    因Resize算子精度问题，修改ONNX模型

    ```
    python3 modify_onnx.py --input_onnx pse/PSE_MobileNetV3.onnx --output pse/modified.onnx

    ```
    参数说明：
    + --input_onnx: 输入的ONNX模型的路径
    + --output: 生成ONNX模型的保存路径

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
        --model=pse/modified.onnx \
        --input_shape="x:${bs},3,736,1312" \
        --output=pse/PSE_MobileNetV3_bs${bs} \
        --input_format=NCHW \
        --log=error \
        --soc_version=Ascend${chip_name} \
        --insert_op_conf=pse_aipp.cfg \
        --enable_small_channel=1
    ```

   参数说明：
    + --model: ONNX模型路径
    + --framework: 5代表ONNX模型
    + --input_shape: 输入数据的shape
    + --input_format: 输入数据的排布格式
    + --output: OM模型路径，无需加后缀
    + --log：日志级别
    + --soc_version: 处理器型号
    + --insert_op_conf: 插入算子的配置文件路径，例如aipp预处理算子
    + --enable_small_channel: 是否使能small channel的优化 

    命令中的`${bs}`表示模型输入的 batchsize，比如将`${bs}`设为 1，则运行结束后会在 pse 目录下生成 PSE_MobileNetV3_bs1.om


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
    mkdir pse/results
    python3 -m ais_bench \
        --model pse/PSE_MobileNetV3_bs1.om \
        --input pse/bin_list/ \
        --output pse/results/ \
        --output_dirname bs1_out/ \
        --batchsize 1
    ```
    参数说明：
    + --model: OM模型路径
    + --input: 存放预处理bin文件的目录路径
    + --output: 存放推理结果的父目录
    + --output_dirname: 存放推理结果的目录，该目录位于--output指定的目录下
    + --batchsize: 模型每次处理样本的数量
    
    运行成功后，每个预处理 bin 文件会对应生成一个推理结果 bin 文件存放在 pse/results/bs1_out/ 目录下。
  
3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的准确率：
    ```shell
    python3 pse_postprocess.py \
        -c configs/det/det_mv3_pse.yml  \
        -o res_dir=pse/results/bs1_out/ info_dir=pse/info_list/
    ```
    参数说明：
    + -c, --config: 模型配置文件路径
    + -o res_dir: 存放推理结果的目录路径
    + -o info_dir: 存放groundtruth等信息的目录路径
    
    运行成功后，程序会打印出模型的精度指标：
    ```
    precision:0.8214486243683324
    recall:0.7043813192103996
    hmean:0.7584240539139452
    ```

4. 性能验证  

    对于性能的测试，需要注意以下三点：
    + 测试前，请通过 npu-smi info  命令查看 NPU 设备状态，请务必在 NPU 设备空闲的状态下进行性能测试。
    + 为避免因测试持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps.

    > 吞吐率（throughput）：模型在单位时间（1秒）内处理的数据样本数。
    
    执行纯推理：
    ```shell
    python3 -m ais_bench --model pse/PSE_MobileNetV3_bs1.om --loop 100 --batchsize 1
    ```

    执行完纯推理命令，程序会打印出与性能相关的指标，则找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。
    
----
# 性能&精度

在310P设备上，OM模型个各batchsize的精度与目标精度[{precision:82.20%, recall:70.48%, hmean:75.89%}](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/algorithm_det_psenet.md#1-%E7%AE%97%E6%B3%95%E7%AE%80%E4%BB%8B)各指标的相对误差均低于 1%，当 batchsize 为 1 时模型性能最优，达 219.49 fps.

各batchsize的精度与性能指标如下：

| 芯片型号 | Batch Size | 数据集 | 精度 | 性能 |
| --------- | -------- | --------- | -------------------------------------------------------- | --------- |
|Ascend310P3| 1        | ICDAR2015 | {'precision': 0.8214, 'recall': 0.7044, 'hmean': 0.7584} | 219.49 fps |
|Ascend310P3| 4        | ICDAR2015 | {'precision': 0.8214, 'recall': 0.7044, 'hmean': 0.7584} | 143.86 fps |
|Ascend310P3| 8        | ICDAR2015 | {'precision': 0.8214, 'recall': 0.7044, 'hmean': 0.7584} | 140.41 fps |
|Ascend310P3| 16       | ICDAR2015 | {'precision': 0.8214, 'recall': 0.7044, 'hmean': 0.7584} | 136.94 fps |
|Ascend310P3| 32       | ICDAR2015 | {'precision': 0.8214, 'recall': 0.7044, 'hmean': 0.7584} | 137.18 fps |
|Ascend310P3| 64       | ICDAR2015 | {'precision': 0.8214, 'recall': 0.7044, 'hmean': 0.7584} | 175.27 fps |


