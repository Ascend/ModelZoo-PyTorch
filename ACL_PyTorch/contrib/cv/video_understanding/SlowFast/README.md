# SlowFast 模型推理指导

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

SlowFast 是用于视频理解的双流框架的卷积神经网络，该网络将视频片段以不同的FPS输入两个由ResNet膨胀而来的神经网络中提取特征。Fast分支接受输入帧更多，通道数更小；Slow分支接受输入帧更少，通道数更大，同时会不断融合Fast分支中的特征。

+ 论文  
    [SlowFast Networks for Video Recognition](https://openaccess.thecvf.com/content_ICCV_2019/html/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.html)  
    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, Kaiming He;   
    Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 6202-6211  

+ 参考实现：  
    url=https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py  
    branch=master  
    commit_id=92e5517f1b3cbf937078d66c0dc5c4ba7abf7a08  

## 输入输出数据
+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | input      | RGB_FP32  | NCHW | bs x 1 x 3 x 32 x 224 x 224 | 

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output1     |  FLOAT32   | ND          | bs x 400   |


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

## 获取源码

1. 安装推理过程所需的依赖
    ```bash
    pip install -r requirements.txt
    ```
2. 获取开源仓源码
    ```bash
    git clone https://github.com/open-mmlab/mmaction2.git
    cd mmaction2
    git checkout 92e5517f1b3cbf937078d66c0dc5c4ba7abf7a08
    git am --signoff < ../slowfast.patch
    pip install -v -e .
    cd ..
    ```

## 准备数据集

1. 获取原始数据集  
    使用 kinetics400 数据集，受磁盘空间限制，slowfast 的离线推理使用 video 格式的数据集（没有抽帧成 rawframes），故需要安装 decord 用于数据前处理时的在线解帧。在 x86 架构下，可以直接使用指令 `pip install decord` 安装，而在 arm 架构下，需源码编译安装 decord。下载 video 格式的数据集，可按照 MMAction2 [准备 Kinetics 数据集](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/kinetics/README_zh-CN.md) 中的引导进行。下载数据集和标注文件并生成文件列表的指令如下所示： 
    ```bash
    cd ./mmaction2/tools/data/kinetics
    
    # 若只下载验证集用于推理，则可以删除以下 3 个 shell 脚本中，与训练集有关的指令
    bash download_annotations.sh kinetics400
    bash download_videos.sh kinetics400
    bash generate_videos_filelist.sh kinetics400
    
    cd ../../../..
    ```
    按如上操作整理数据集和文件列表后，文件树应当如下:  
    ```
    ├── mmaction2
        ├── data
            ├── kinetics400
                ├── kinetics400_val_list_videos.txt
                ├── videos_val/
    ```


2. 数据预处理  
    执行前处理脚本将原始数据转换为OM模型输入需要的bin/npy文件。
    ```bash
    python slowfast_preprocess.py \
        --config mmaction2/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py \
        --video_dir mmaction2/data/kinetics400/videos_val \
        --ann_file mmaction2/data/kinetics400/kinetics400_val_list_videos.txt \
        --save_dir prep_dataset
    ```
    参数说明：
    + --config: 模型配置文件路径
    + --video_dir: 测试视频所在的目录路径
    + --ann_file: 标签文件所在路径
    + --save_dir: 存放生成的bin文件的目录路径
    
    运行成功后，每个测试视频都会对应生成一个bin文件存放于`./prep_dataset/bin`目录下。此外还会生成一个`./prep_dataset/kinetics400.info`文件，用于记录每个bin文件的label。


## 模型转换

1. PyTroch 模型转 ONNX 模型  
 
    下载open-mmlab官方提供的[ **预训练模型** ](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth) 到当前目录，可参考命令：
    ```bash
    wget https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth
    ```

    然后执行执行以下命令生成 ONNX 模型：
    ```bash
    python slowfast_pth2onnx.py \
        --config mmaction2/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py \
        --checkpoint slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth \
        --output-file slowfast.onnx \
        --softmax
    ```
    参数说明：
    + --config: 模型配置文件路径
    + --checkpoint: 预训练权重文件的路径。
    + --output-file: 生成ONNX模型的保存路径
    + --softmax: 是否在识别器末端加上加上softmax

2. ONNX模型优化
    ```bash
    python -m onnxsim slowfast.onnx slowfast_sim.onnx
    ```
    slowfast.onnx为原始的ONNX模型，经过onnx-simplifier优化后，生成slowfast_sim.onnx。

3. ONNX 模型转 OM 模型  

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
    bs=1  # 根据需要自行设置batchsize

    
    # 执行 ATC 进行模型转换
    atc --model=slowfast_sim.onnx \
        --framework=5 \
        --output=slowfast_bs${bs} \
        --input_format=ND \
        --input_shape="video:${bs},1,3,32,224,224" \
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
    python -m ais_bench \
        --model slowfast_bs${bs}.om \
        --input ./prep_dataset/bin/ \
        --output ./ \
        --output_dirname result_bs${bs} \
        --outfmt TXT \
        --batchsize ${bs}
    ```
    参数说明：
    + --model OM模型路径
    + --input 存放预处理后数据的目录路径
    + --output 用于存放推理结果的父目录路径
    + --output_dirname 用于存放推理结果的子目录名，位于--output指定的目录下
    + --outfmt 推理结果文件的保存格式
    + --batchsize 模型每次输入bin文件的数量


2. 性能验证  
    对于性能的测试，需要注意以下三点：
    + 测试前，请通过`npu-smi info`命令查看NPU设备状态，请务必在NPU设备空闲的状态下进行性能测试。
    + 为了避免测试过程因持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps，反映模型在单位时间（1秒）内处理的样本数。
    ```bash
    python -m ais_bench --model slowfast_bs${bs}.om --batchsize ${bs} --loop 100
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python slowfast_postprocess.py --result_dir ./result_bs${bs}/  --label_file prep_dataset/kinetics400.info
    ```
    参数说明：
    + --result_dir: 存放推理结果的目录路径
    + --label_file: 数据预处理生成的标签文件路径
    
    运行成功后，程序会将会打印出模型的精度指标：
    ```
    Evaluating top_k_accuracy ...

    top1_acc        0.7007
    top5_acc        0.8855
    ```


----
# 性能&精度

在310P设备上，当batchsize设为1时OM模型性能最优，达 146.4 fps，此时模型精度为  **{Top1@Acc=70.07%, Top5@Acc=88.55%}**

| 芯片型号   | BatchSize | 数据集      | 精度                              | 性能      |
| --------- | --------- | ----------- | -------------------------------- | --------- |
|Ascend310P3| 1         | kinetics400 | Top1@Acc=70.07%, Top5@Acc=88.55% | 138.4 fps |
|Ascend310P3| 4         | kinetics400 |                                  | 131.1 fps |
|Ascend310P3| 8         | kinetics400 | Top1@Acc=70.08%, Top5@Acc=88.55% | 128.1 fps |
|Ascend310P3| 16        | kinetics400 |                                  | 128.8 fps |
|Ascend310P3| 32        | kinetics400 |                                  | 129.0 fps |
|Ascend310P3| 64        | kinetics400 |                                  | 129.5 fps |
