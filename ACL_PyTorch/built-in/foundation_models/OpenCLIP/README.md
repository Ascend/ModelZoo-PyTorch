# OpenCLIP模型推理指导

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

[OpenCLIP](https://github.com/mlfoundations/open_clip.git) 是 OpenAI [CLIP](https://github.com/openai/CLIP) 的开源实现，作者团队将[原 CLIP 论文](https://arxiv.org/abs/2103.00020)中训练数据的规模从 400M 提升到 2B，在39个数据集（CLIP_benchmark）上完整的实验评估结果证明：扩增定律 (Scaling Laws) 同样适用于 CLIP 模型。

+ 论文  

    [Reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143)  
    Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, Jenia Jitsev

+ 参考实现

    url = https://github.com/mlfoundations/open_clip.git  
    version = v2.20.0  
    commit_id = a09d519f91358187d73debacfc2db5f154291d40  
    model_name = ViT-32-B  

## 输入输出数据

CLIP 模型接受图像与文本双模态输入，图像与文本是一对多的数量关系，即一张图片，对应N条描述该图片的文本信息，N的具体数值取决于使用的数据集。在离线推理时，两个模态单独转出模型，其输入输出详情如下：

+ 视觉模型
    |      | 节点名      | 数据类型 | 数据排布 | 数据尺寸 |
    | ---- | ----------- | ------- | ------- | -------- |
    | 输入 | image       | FLOAT32 | NCHW     | bs x 3 x 224 x 224 |
    | 输出 | image_embed | FLOAT32 | NCHW     | bs x 512 |

+ 文本模型
    |      | 节点名     | 数据类型    | 数据排布 | 数据尺寸 |
    | ---- | --------- | ---------- | -------- | -------- |
    | 输入 | text       |  INT64     | ND      | bs x 77  |
    | 输出 | text_embed |  FLOAT32   | ND      | bs x 512 |


----
# 推理环境

- 该模型推理所需配套的软件如下：  

    | 配套      | 版本    | 环境准备指导 |
    | --------- | ------- | ---------- |
    | 固件与驱动 | 1.0.20.alpha  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN      | 7.0.RC1   | -          |
    | Python    | 3.8.13     | -          |

    说明：请根据推理卡型号与 CANN 版本选择相匹配的固件与驱动版本。


----
# 快速上手

## 获取源码

1. 安装推理过程所需的依赖
    ```bash
    pip3 install -r requirements.txt
    ```
2. 获取开源仓源码
    ```bash
    git clone https://github.com/mlfoundations/open_clip.git
    cd open_clip
    git checkout v2.20.0
    pip3 install -v -e .
    cd ..
    ```

## 准备数据集

1. 获取原始数据集  

    该模型使用 [flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) 数据集来验证模型精，这里沿用 CLIP_benchmark 提供的[测试集](https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/flickr30k_test_karpathy.txt)，该测试集包含 1K 测试图片以及对应的 5K 文本描述信息，自行下载，然后按照以下的方式存放：

    ```
    ├── ./flickr30k_test1k/
        ├── 1007129816.jpg
        ├── 1009434119.jpg
        ├── 101362133.jpg
        ├── ...
        └── flickr30k_test_karpathy.txt  # 文本信息，5000行
    ```

2. 数据预处理  

    执行以下命令将原始图片和原始文本处理成模型可接受的输入：
    ```bash
    python3 openclip_preprocess.py --data-dir ./flickr30k_test1k --save-dir ./prep_data
    ```
    + 参数说明：
        + --data-dir: 测试数据所在的目录
        + --save-dir: 预处理后的数据存放目录

    运行结束后，预处理数据存放目录的结构如下：
    ```
    ├── ./prep_data/
        └── images/  # 包含1000个npy文件
            ├── 1007129816.npy
            ├── ...
            └── 97234558.npy
        └── texts/   # 包含5000个npy文件
            ├── 1007129816_0.npy
            ├── ...
            └── 97234558_4.npy
    ```

## 模型转换

1. PyTroch 模型转 ONNX 模型  

    执行以下命令，导出视觉模型与文本模型的ONNX：
    ```bash
    mkdir models
    python3 openclip_torch2onnx.py --model-name ViT-B-32 --pretrained laion2b_s34b_b79k --onnx-prefix models/vit_b_32
    ```
    + 参数说明：
        + --model-name: 模型名字
        + --pretrained: 指定预训练权重名，程序会自动下载权重文件，也可以直接指定下载好的权重文件的路径。
        + --onnx-prefix: 保存ONNX模型时的前缀

    运行时，此程序会自动下载模型的预训练权重，运行结束后，`models` 目录下将会生成 `vit_b_32_vision.onnx`，`vit_b_32_text.onnx`。
    
    如果自动下载权重失败导致程序终止，可以进入该[模型主页](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)，手动下载文件`open_clip_pytorch_model.bin` 并将其放到当前目录下，然后导 ONNX 时指通过`--pretrained`参数指定权重文件路径：
    ```bash
    python3 openclip_torch2onnx.py --model-name ViT-B-32 --pretrained ./open_clip_pytorch_model.bin --onnx-prefix models/vit_b_32
    ```


2. ONNX 模型转 OM 模型  

    step1: 查看NPU芯片名称 \${chip_name}
    ```bash
    npu-smi info
    ```
    假如该设备芯片名为 310P3，则回显如下：
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

    chip_name=910B4
    bs=1

    # 生成视觉 OM 模型
    atc --framework=5 \
        --model=models/vit_b_32_vision.onnx \
        --output=models/vit_b_32_vision_bs${bs} \
        --input_format=NCHW \
        --input_shape="image:${bs},3,224,224" \
        --log=error \
        --soc_version=Ascend${chip_name}

    # 生成文本 OM 模型
    atc --framework=5 \
        --model=models/vit_b_32_text.onnx \
        --output=models/vit_b_32_text_bs${bs} \
        --input_format=ND \
        --input_shape="text:${bs},77" \
        --log=error \
        --soc_version=Ascend${chip_name}
    ```

   ATC 参数说明：  

    + --framework: 5 代表 ONNX 模型
    + --model: ONNX 模型路径
    + --input_shape: 模型输入数据的 shape
    + --input_format: 输入数据的排布格式
    + --output: OM 模型路径，无需加后缀
    + --log：日志级别
    + --soc_version: 处理器型号


## 推理验证

1. 对数据集推理  

    该离线模型使用ais_bench作为推理工具，请参考ais_bench的[**Gitee主页**](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)安装推理后端包aclruntime与推理前端包ais_bench。  
    可直接使用命令方式对预处理后的数据进行推理，参考命令：
    ```bash
    # 视觉模型推理，生成测试图片的视觉 embeddings
    python3 -m ais_bench \
        --model models/vit_b_32_vision_bs${bs}.om \
        --input prep_data/images \
        --output ./om_outputs \
        --output_dirname vision_embed_bs${bs} \
        --outfmt NPY

    # 文本模型推理，生成测试图片的文本 embeddings
    python3 -m ais_bench \
        --model models/vit_b_32_text_bs${bs}.om \
        --input prep_data/texts \
        --output ./om_outputs \
        --output_dirname text_embed_bs${bs} \
        --outfmt NPY
    ```
    参数说明：
    + --model: OM 模型路径
    + --input: 存放预处理后数据的目录路径
    + --output: 用于存放推理结果的父目录路径
    + --output_dirname: 用于存放推理结果的子目录名，位于`--output`指定的目录下
    + --outfmt: 推理结果的保存格式

2. 精度验证  

    使用后处理脚本计算模型的各精度指标：
    ```bash
    python3 openclip_postprocess.py \
        --vision-embeds om_outputs/vision_embed_bs${bs} \
        --text-embeds om_outputs//text_embed_bs${bs}
    ```
    参数说明：
    + --vision-embeds: 视觉模型推理结果保存目录
    + --text-embeds: 文本模型推理结果保存目录

    运行结束后，程序将会打印出模型在 flickr30k 数据集上做 Zero-Shot Retrieval 任务的精度指标：
    ```
    Metrics: {'image_retrieval_recall@5': 0.8841999769210815, 'text_retrieval_recall@5': 0.962000012397766}
    ```

3. 性能验证

    对于性能的测试，需要注意以下三点：
    + 测试前，请通过`npu-smi info`命令查看NPU设备状态，请务必在NPU设备空闲的状态下进行性能测试。
    + 为了避免测试过程因持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps，反映模型在单位时间（1秒）内处理的样本数。
    ```bash
    python3 -m ais_bench --model models/vit_b_32_vision_bs${bs}.om --loop 100 --batchsize ${bs}
    python3 -m ais_bench --model models/vit_b_32_text_bs${bs}.om --loop 100 --batchsize ${bs}
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。


----
# 性能&精度

该模型在 Ascend910B4 设备上进行测试，测得 OM 模型在 flickr30k 数据集上的精度指标如下：

| 指标                     | OM实测 | PyTorch实测 |
| ------------------------ | ------ | ----------- |
| image_retrieval_recall@5 | 0.8841 | 0.8830     |
| text_retrieval_recall@5  | 0.9620 | 0.9630     |

测得 OM 在 Ascend910B4 设备上纯推理性能如下：

| BatchSize | 视觉模型性能 (fps) | 文本模型性能 (fps) |
| --------- | ----------------- | ----------------- |
| 1         | 181               | 407               |
| 4         | 251               | 1031              |
| 8         | 285               | 1944              |
| 16        | 306               | 2999              |
| 32        | 319               | 4139              |
| 64        | **326**           | **4502**          |
