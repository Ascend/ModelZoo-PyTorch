# Dino_Resnet50模型-推理指导

- [概述](#概述)
- [推理环境](#推理环境)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型转换](#模型转换)
  - [推理验证](#推理验证)
- [精度&性能](#精度性能)

---

# 概述

Dino是Facebook于今年发表的最新的无监督学习成果，在图像处理分类等方面取得了很好的成果，而与经典的Resnet50的分类模型的残差单元相结合训练，经验证也依然保障了较高精度，与纯Resnet50模型相比精度基本没有下滑，同时也保持了性能。

- 论文  
    [Caron, Mathilde, et al. "Emerging properties in self-supervised vision transformers." arXiv preprint arXiv:2104.14294 (2021).](https://arxiv.org/abs/2104.14294)
    

- 参考实现

    ```
    url = https://github.com/facebookresearch/dino
    branch = main
    commit_id = cb711401860da580817918b9167ed73e3eef3dcf 
    ```

- 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | input | FP32 | NCHW | 1 x 3 x 224 x 224 |

- 模型输出  
    | output-name | data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output | FLOAT32 | ND | 1 x 1000 |

    
---

# 推理环境

- 该模型需要以下插件与驱动

    | 配套     | 版本          | 环境准备指导                                                                                           |
    | -------- | ------------- | ---------------------------------------------------------------------------------------------------- |
    | 固件与驱动 | 22.0.2 | [Pytorch 框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN     | 5.1.RC2       | -                                                                                                  |
    | Python   | 3.7.5         | -                                                                                                     |

---

# 快速上手

## 获取源码


1. 下载本仓，复制该推理项目所在目录，进入复制好的目录
    ```
    cd Dino
    ```

2. 执行以下命令安装所需的依赖
    ```shell
    pip install -r requirements.txt
    ```

3. 克隆开源仓并修改源码
    ```shell
    git clone https://github.com/facebookresearch/VideoPose3D.git
    cd VideoPose3D
    git checkout main
    git reset --hard 1afb1ca0f1237776518469876342fc8669d3f6a9
    patch -p1 < ../vp3d.patch
    cd ..
    ```

4. 创建一个目录，用于存放整个推理过程中所需文件与生成文件
    ```shell
    mkdir vp3d
    ```

## 准备数据集

1. 获取原始数据集  
    本模型使用 [ImageNet官网](http://www.image-net.org) 的5万张验证集进行测试，以ILSVRC2012为例，本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。
    ```
    最终，数据的目录结构如下：
    ```
    ├── imageNet
       ├── val
       └── val_label.txt
    ```

2. 数据预处理  
    运行数据预处理脚本，将原始数据转换为符合模型输入要求的bin文件。
    ```shell
    python3.7 dino_resnet50_preprocess.py resnet ${datasets_path}/imagenet/val ${prep_output_dir}
    ```
    参数说明：
    + “resnet”：数据预处理方式为resnet网络。
    + ${datasets_path}/imagenet/val: 原始数据验证集（.jpeg）所在路径。
    + ${prep_output_dir}: 输出的二进制文件（.bin）所在路径。
    
    运行成功后，会在当前目录下生成二进制文件。


## 模型转换

1. PyTroch 模型转 ONNX 模型  
    step1 获取权重文件  
    该推理项目使用源码包中的权重文件（dino_resnet50_pretrain.pth和dino_resnet50_linearweights.pth）。

    step2 导出 .onnx 文件
    ```
    python3.7 dino_resnet50_pth2onnx.py
    ```

    
2. ONNX 模型转 OM 模型
    step1: 查看 NPU 芯片名称 \${chip_name}

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
    atc --framework=5 --model=dino_resnet50.onnx --output=dino_resnet50_bs1 --input_format=NCHW --input_shape="input:1,3,224,224" --log=debug --soc_version=${chip_name}
    ```
    
    参数说明：
    + --model: ONNX模型文件所在路径。
    + --framework: 5 代表ONNX模型。
    + --input_format: 输入数据的排布格式。
    + --input_shape: 输入数据的shape。
    + --output: 生成OM模型的保存路径。
    + --log: 日志级别。
    + --soc_version: 处理器型号。
    
    运行成功后，在当前目录下会生成名为 dino_resnet50_bs1.om 的模型文件。

## 推理验证

1. 准备推理工具

    本推理项目使用 [ais_infer](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer#%E4%BB%8B%E7%BB%8D) 作为推理工具，须自己拉取源码，打包并安装。
    
    ```shell
    # 指定CANN包的安装路径
    export CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest
    
    # 获取推理工具源码
    git clone https://gitee.com/ascend/tools.git
    cp -r tools/ais-bench_workload/tool/ais_infer .
    
    # 打包
    cd ais_infer/backend/
    pip3 wheel ./   # 会在当前目录下生成 aclruntime-xxx.whl，具体文件名因平台架构而异
    
    # 安装
    pip3 install --force-reinstall aclruntime-xxx.whl
    cd ../..
    ```

2. 离线推理

    使用 ais_infer 工具将预处理后的数据传入模型并执行推理：
    ```shell
    # 设置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /etc/profile
    
    # 对预处理后的数据进行推理
    mkdir dino/infer_results/
    python3 ais_infer/ais_infer.py \
        --model "dino_resnet50_bs1.om" \
        --input "{prep_output_dir}/" \
        --output "dino/infer_results/" \
        --batchsize 1
        --outfmt TXT
    ```
    参数说明：
    + --model: OM模型路径。
    + --input: 存放预处理bin文件的目录路径
    + --output: 存放推理结果的目录路径
    + --batchsize：每次输入模型的样本数
    + --outfmt: 推理结果数据的格式
    运行成功后，在 dino/infer_results/ 下，会生成一个以执行开始时间%Y_%m_%d-%H_%M_%S来命名的子目录，每个预处理 bin 文件会对应生成一个推理结果 txt 文件存放在此目录下。

3. 精度验证

    执行后处理脚本，根据推理结果与 label比对 计算 OM 模型的准确率：
    ```shell
    python3.7 dino_resnet50_postprocess.py --anno_file ${datasets_path}/val_label.txt --ais_infer ./result/2022_10_18-11_26_11 --result_file ./result.json
    ```
    
    参数说明：
    + --anno_file: 标签数据位置
    + --ais_infer: 推理结果所在路径
    + --result_file: 输出结果位置
    说明：精度验证之前，将推理结果文件中summary.json删除
    运行成功后，程序会打印出模型的精度指标：
    ```
    ==== Validation Results ====
    {"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "75.28%"}, {"key": "Top2 accuracy", "value": "85.38%"}, {"key": "Top3 accuracy", "value": "89.24%"}, {"key": "Top4 accuracy", "value": "91.31%"}, {"key": "Top5 accuracy", "value": "92.56%"}]}
    ```

4. 性能验证

    对于性能的测试，需要注意以下三点：
    + 测试前，请通过 npu-smi info  命令查看 NPU 设备状态，请务必在 NPU 设备空闲的状态下进行性能测试。
    + 为避免因测试持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps.

    > 吞吐率（throughput）：模型在单位时间（1秒）内处理的数据样本数。
    
    step1 执行纯推理：
    ```shell
    python3 ais_infer/ais_infer.py --model dino_resnet50_bs1.om --loop 100 --batchsize 1
    ```

    执行完纯推理命令，程序会打印出与性能相关的指标，找到 **NPU_compute_time** 中的 **mean** 字段，其含义为推理的平均耗时，单位为毫秒(ms)。

---

# 精度&性能

1. 精度对比

    | Model       | batchsize | Accuracy | 开源仓精度 |
    | ----------- | --------- | -------- | ---------- |
    | dino_resnet50 | 1       | top1 accuracy = 75.28% top5 accuracy = 92.56% | top1 accuracy = 75.28% top5 accuracy = 92.56%|
    | dino_resnet50| 16      | top1 accuracy = 75.28% top5 accuracy = 92.56% | top 1 accuracy = 75.28% top5 accuracy = 92.56%|
2. 性能对比
    | batchsize | 310 性能 | T4 性能 | 310P 性能 | 310P/310 | 310P/T4 |
    | ---- | ---- | ---- | ---- | ---- | ---- |
    | 1 | 1617.052 fps | 878.742 fps | 1378.7 fps | 0.85 | 1.6 |
    | 4 | 2161.044 fps | 1532.6 fps   | 5539.4 fps |2.5|  3.6|
    | 8 | 2410.1 fps     | 1733.5fps    | 10986 fps |4.5| 6.3|
    | 16| 2441.2 fps  |  1858.1fps    |  22119 fps |9  |  11|
    | 32 | 5279.8fps | 2033.1fps    | 43852fps |    8 |21.5|
    | 64| 2244fps  |2090fps | 87537fps|     39| 41|