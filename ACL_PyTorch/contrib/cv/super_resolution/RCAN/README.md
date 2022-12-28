# RCAN 模型推理指导

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

RCAN设计了一个残差中的残差（RIR）结构来构造深层网络，每个 RIR 结构由数个残差组（RG）以及长跳跃连接（LSC）组成，每个 RG 则包含一些残差块和短跳跃连接（SSC）。RIR 结构允许丰富的低频信息通过多个跳跃连接直接进行传播，使主网络专注于学习高频信息。此外，我们还提出了一种通道注意力机制（CA），通过考虑通道之间的相互依赖性来自适应地重新调整特征。解决了过深的网络却难以训练。低分辨率的输入以及特征包含丰富的低频信息，但却在通道间被平等对待，因此阻碍了网络的表示能力的问题。

+ 论文  
    [Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://arxiv.org/abs/1807.02758)  
    Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, Yun Fu  

+ 参考实现：  
    url = https://github.com/yulunzhang/RCAN  
    branch = master  
    commit_id = 3339ebc59519c3bb2b5719b87dd36515ec7f3ba7  

## 输入输出数据
+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | input      | RGB_FP32  | NCHW        | batchsize x 3 x 256 x 256 | 

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output1     |  RGB_FP32  | NCHW        | batchsize x 3 x 512 x 512   |


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
    git clone https://github.com/yulunzhang/RCAN.git -b master
    cd RCAN
    git checkout 3339ebc59519c3bb2b5719b87dd36515ec7f3ba7
    cd ..
    ```

## 准备数据集

1. 获取原始数据集  
    该模型使用 [Set5](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html) 的5张验证集图片进行精度验证，可前往 [Hugging Face](https://huggingface.co/datasets/eugenesiow/Set5/tree/main/data) 自行下载`Set5_HR.tar.gz`和`Set5_LR_x2.tar.gz`，然后将两个压缩包内的图片分别解压到`Set5/HR`和`Set5/LR`目录。
    按以上操作获取数据集并解压后，数据的目录结构如下:  
    ```
    ├── Set5/
        ├── HR/
            ├── baby.png
            ├── bird.png
            ├── butterfly.png
            ├── head.png
            └── woman.png
        └── LR/
            ├── baby.png
            ├── bird.png
            ├── butterfly.png
            ├── head.png
            └── woman.png
    ```


2. 数据预处理  
    执行前处理脚本将原始数据转换为OM模型输入需要的bin/npy文件。
    ```bash
    python rcan_preprocess.py -s ./Set5/LR -o ./prep_data -sz 256
    ```
    参数说明：
    + -s/--source: 原始数据路径
    + -o/--output: 保存输出bin文件的目录路径
    + -sz/--size: 统一大小后的尺寸，默认为256
    
    预处理程序会对原始图片进行pad和缩放操作，从而将不同shape的图片处理成同一大小。上述命令运行结束后，`./prep_data`目录下会生成一个pad_info.json文件来记录在预处理中图片的pad和缩放信息，用于后处理时进行图像裁剪。此外，`./prep_data`目录下目录下还会生成一个名为`bin`的子目录，用于存放预处理后生成的bin文件。


## 模型转换

1. PyTroch 模型转 ONNX 模型  
 
    进入 [Dropbox](https://www.dropbox.com/s/qm9vc0p0w9i4s0n/models_ECCV2018RCAN.zip?dl=0) / [BaiduYun](https://pan.baidu.com/s/1bkoJKmdOcvLhOFXHVkFlKA) / [GoogleDrive](https://drive.google.com/file/d/10bEK-NxVtOS9-XSeyOZyaRmxUTX3iIRa/view?usp=sharing) 任意一个下载通道，下载开源仓提供的预训练权重到当前目录，解压缩。该推理任务只需用到 `models_ECCV2018RCAN/RCAN_BIX2.pt`。可通过md5sum值(f567f8560fde71ba0973a7fe472a42f2)来检查预训练权重文件的完整性。

    然后执行执行以下命令生成 ONNX 模型：
    ```bash
    python rcan_pth2onnx.py --pth ./models_ECCV2018RCAN/RCAN_BIX2.pt --onnx ./rcan.onnx --shape 256 256 --scale 2
    ```
    参数说明：
    + --pth: 预训练权重文件的路径
    + --onnx: 生成ONNX模型的保存路径
    + --shape: 模型输入数据的形状，须与预处理时的--size保持一致
    + --scale: 模型输出图片相对于输入图片的放大倍数，默认为2

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
    bs=1  # 根据需要自行设置batchsize

    # 执行 ATC 进行模型转换
    atc --framework=5 \
        --model=rcan.onnx \
        --output=rcan_bs${bs} \
        --input_format=NCHW \
        --input_shape="image:${bs},3,256,256" \
        --log=debug \
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
    python -m ais_bench \
        --model rcan_bs${bs}.om \
        --input ./prep_data/bin/ \
        --output ./ \
        --output_dirname result_bs${bs} \
        --batchsize ${bs}
    ```
    参数说明：
    + --model OM模型路径
    + --input 存放预处理后数据的目录路径
    + --output 用于存放推理结果的父目录路径
    + --output_dirname 用于存放推理结果的子目录名，位于--output指定的目录下
    + --batchsize 模型每次输入bin文件的数量


2. 性能验证  
    对于性能的测试，需要注意以下三点：
    + 测试前，请通过`npu-smi info`命令查看NPU设备状态，请务必在NPU设备空闲的状态下进行性能测试。
    + 为了避免测试过程因持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps，反映模型在单位时间（1秒）内处理的样本数。
    ```bash
    python -m ais_bench --model rcan_bs${bs}.om --batchsize ${bs} --loop 100
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python rcan_postprocess.py \
        --infer_results ./result_bs${bs} \
        --pad_info ./prep_data/pad_info.json \
        --hr_images ./Set5/HR \
        --save_dir ./gen_images_bs${bs} \
        --shape 256 256 \
        --scale 2
    ```
    参数说明：
    + --infer_results: 存放推理结果的目录路径
    + --pad_info: 数据预处理生成的pad信息文件路径
    + --hr_images: 存放原始HR图片的目录路径
    + --save_dir: 模型输出经后处理后，生成图片的保存目录
    + --shape: 模型输入数据的形状，须与预处理时的--size保持一致
    + --scale: 模型输出图片相对于输入图片的放大倍数，默认为2，须跟导出ONNX时的--scale参数保持一致
    
    运行成功后，程序会根据推理结果生成放大后的图片，并打印出模型的精度指标：
    ```
    Images generated! path: ./gen_images_bs1
    Evaluation of RCAN model
    PSNR    38.249656290876096
    SSIM    0.9606179588265406
    ```

----
# 性能&精度

在310P设备上，模型精度为  **{PSNR=38.25, SSIM=0.9606}**，当batchsize设为 1 时OM模型性能最优，达 **12.25 fps**。

| 芯片型号   | BatchSize | 数据集      | 精度                    | 性能       |
| --------- | --------- | ----------- | ----------------------- | --------- |
|Ascend310P3| 1         | Set5        | PSNR=38.25, SSIM=0.9606 | **12.25 fps** |
|Ascend310P3| 4         | Set5        | PSNR=38.25, SSIM=0.9606 | 10.39 fps |
|Ascend310P3| 8         | Set5        | PSNR=38.25, SSIM=0.9606 | 11.08 fps |
|Ascend310P3| 16        | Set5        | PSNR=38.25, SSIM=0.9606 | 11.21 fps |
|Ascend310P3| 32        | Set5        | PSNR=38.25, SSIM=0.9606 | 11.37 fps |
|Ascend310P3| 64        | Set5        | PSNR=38.25, SSIM=0.9606 | 10.96 fps |
