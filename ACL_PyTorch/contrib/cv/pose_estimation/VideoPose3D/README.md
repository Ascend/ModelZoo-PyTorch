# VideoPose3D 模型推理指导

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

VideoPose3D 是一个基于时间维度上膨胀卷积的高效全卷积网络。它能够仅利用 2D 视频的姿态估计结果，通过时间维度上的卷积网络高效地推断出相应地 3D 姿态。与基于 RNN 的同类任务网络相比，VideoPose3D 不仅在效率上更高，而且在精度上也更好。实现在参数和计算成本更少的情形下比此前所有网络更优的性能。

- 论文  
    [3D human pose estimation in video with temporal convolutions and semi-supervised training](https://arxiv.org/pdf/1811.11742.pdf)
    [Dario Pavllo](https://arxiv.org/search/cs?searchtype=author&query=Pavllo%2C+D), [Christoph Feichtenhofer](https://arxiv.org/search/cs?searchtype=author&query=Feichtenhofer%2C+C), [David Grangier](https://arxiv.org/search/cs?searchtype=author&query=Grangier%2C+D), [Michael Auli](https://arxiv.org/search/cs?searchtype=author&query=Auli%2C+M)
    

- 参考实现

    ```
    url = https://github.com/facebookresearch/VideoPose3D.git
    branch = main
    commit_id = 1afb1ca0f1237776518469876342fc8669d3f6a9
    ```

- 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | input | FP32 | NCHW | 2 x 6115 x 17 x 2 |

- 模型输出  
    | output-name | data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output | FLOAT32 | NCHW | 2 x 6115 x 17 x 3 |

    注：输入、输出数据的格式中，第一位N=2，是因为VideoPose3D在推理阶段采取了test-time data augmentation。它的数据是人体姿态，因此正向推理一遍，左右镜像后再推理一遍。将两遍结果融合后，得到最终结果，因此N=2，因为它表示正、反两组数据。
---

# 推理环境

- 该模型离线推理使用 Atlas 300I Pro 推理卡，推理所需配套的软件如下：

    | 配套     | 版本          | 环境准备指导                                                                                           |
    | -------- | ------------- | ---------------------------------------------------------------------------------------------------- |
    | firmware | 1.82.22.2.220 | [Pytorch 框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | driver   | 22.0.2        | [Pytorch 框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN     | 5.1.RC2       | [Pytorch 框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | Python   | 3.7.5         | -                                                                                                     |

---

# 快速上手

## 获取源码


1. 下载本仓，复制该推理项目所在目录，进入复制好的目录
    ```
    cd VideoPose3D
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
    本模型使用 [Human3.6M](http://vision.imar.ro/human3.6m/description.php) 数据集验证模型精度。参照 [Dataset setup](https://github.com/facebookresearch/VideoPose3D/blob/main/DATASETS.md)，执行以下命令获取原始数据集：
    ```shell
    mkdir vp3d/Human3.6M/
    cd VideoPose3D/data/
    wget https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip
    python prepare_data_h36m.py --from-archive h36m.zip
    mv data_3d_h36m.npz ../../vp3d/Human3.6M/
    cd ../../
    wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_cpn_ft_h36m_dbb.npz -P vp3d/Human3.6M/
    ```
    最终，数据的目录结构如下：
    ```
    ├── vp3d/Human3.6M/
       ├── data_2d_h36m_cpn_ft_h36m_dbb.npz
       └── data_3d_h36m.npz
    ```

2. 数据预处理  
    运行数据预处理脚本，将原始数据转换为符合模型输入要求的bin文件。
    ```shell
    python vp3d_preprocess.py -d vp3d/Human3.6M/ -s vp3d/prep_dataset/
    ```
    参数说明：
    + -d, --dataset: 原始数据验证集（.npz）所在路径。
    + -s, --save: 输出的二进制文件（.bin）保存路径。
    
    运行成功后，会在vp3d/prep_dataset/目录下生成delta_dict_padding.json文件、ground_truths/目录和inputs/目录。其中 inputs目录下存放的是供模型推理的bin文件。


## 模型转换

1. PyTroch 模型转 ONNX 模型  
    step1 获取权重文件  
    该推理项目使用自己训练好的权重文件 [model_best.bin](https://ascend-pytorch-model-file.obs.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/pose_estimation/videopose3d/310P/models/model_best.bin)，自行下载并将权重文件放置在 vp3d/ 目录下。

    step2 导出 .onnx 文件
    ```
    python vp3d_pth2onnx.py -m vp3d/model_best.bin -o vp3d/vp3d.onnx
    ```
    参数说明：
    + -m, --model: 预训练权重所在路径
    + -o, --onnx: 生成ONNX模型的保存路径
    
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
    atc --framework=5 \
        --model=vp3d/vp3d.onnx \
        --output=vp3d/vp3d_seq6115 \
        --input_format=NCHW \
        --input_shape="2d_poses:2,6115,17,2" \
        --log=error \
        --soc_version=Ascend${chip_name}
    ```
    
    参数说明：
    + --model: ONNX模型文件所在路径。
    + --framework: 5 代表ONNX模型。
    + --input_format: 输入数据的排布格式。
    + --input_shape: 输入数据的shape。
    + --output: 生成OM模型的保存路径。
    + --log: 日志级别。
    + --soc_version: 处理器型号。
    
    运行成功后，在 vp3d 目录下会生成名为 vp3d_seq6115.om 的模型文件。

## 推理验证

1. 安装ais_bench推理工具  
    
    请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

2. 离线推理

    使用 ais_bench 推理工具将预处理后的数据传入模型并执行推理：
    ```shell
    # 设置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /etc/profile
    
    # 对预处理后的数据进行推理
    mkdir vp3d/infer_results/
    python3 -m ais_bench \
        --model "vp3d/vp3d_seq6115.om" \
        --input "vp3d/prep_dataset/inputs/" \
        --output "vp3d/infer_results/" \
        --batchsize 1
    ```
    参数说明：
    + --model: OM模型路径。
    + --input: 存放预处理bin文件的目录路径
    + --output: 存放推理结果的目录路径
    + --batchsize：每次输入模型的样本数
    
    运行成功后，在 vp3d/infer_results/ 下，会生成一个以执行开始时间%Y_%m_%d-%H_%M_%S来命名的子目录，每个预处理 bin 文件会对应生成一个推理结果 bin 文件存放在此目录下。

3. 精度验证

    执行后处理脚本，根据推理结果与 groudtruth 计算 OM 模型的准确率：
    ```shell
    python vp3d_postprocess.py --preprocess-data vp3d/prep_dataset --infer-results vp3d/infer_results/2022_09_25-07_41_01/
    ```
    
    参数说明：
    + --preprocess-data: 经预处理后的数据集路径
    + --infer-results: 推理结果所在路径
    
    运行成功后，程序会打印出模型的精度指标：
    ```
    ==== Validation Results ====
    Protocol #1 (MPJPE) action-wise average:46.6mm
    ```

4. 性能验证

    对于性能的测试，需要注意以下三点：
    + 测试前，请通过 npu-smi info  命令查看 NPU 设备状态，请务必在 NPU 设备空闲的状态下进行性能测试。
    + 为避免因测试持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
    + 使用吞吐率作为性能指标，单位为 fps.

    > 吞吐率（throughput）：模型在单位时间（1秒）内处理的数据样本数。
    
    step1 执行纯推理：
    ```shell
    python3 -m ais_bench --model vp3d/vp3d_seq6115.om --loop 100 --batchsize 1
    ```

    执行完纯推理命令，程序会打印出与性能相关的指标，找到 **NPU_compute_time** 中的 **mean** 字段，其含义为推理的平均耗时，单位为毫秒(ms)。每次输入模型的数据量为 2 * 6115，可算得模型的吞吐率为：
    $$throughput =\frac{2 * 6115}{mean} * 1000 $$

---

# 精度&性能

1. 精度对比

    | Model       | batchsize | Accuracy | 开源仓精度 |
    | ----------- | --------- | -------- | ---------- |
    | VideoPose3D | 2         | MPJPE = 46.6mm | MPJPE = 46.8mm |

2. 性能对比
    | batchsize | 310 性能 | T4 性能 | 310P 性能 | 310P/310 | 310P/T4 |
    | ---- | ---- | ---- | ---- | ---- | ---- |
    | 2 | 715495 fps | 604554 fps | 280257 fps | 0.39 倍 | 0.46 倍 |

    注：性能不达标，但已通过性能评审。