# Inception_v3 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

GoogLeNet对网络中的传统卷积层进行了修改，提出了被称为Inception的结构，用于增加网络深度和宽度，提高深度神经网络性能。从Inception V1到Inception V4有4个更新版本，每一版的网络在原来的基础上进行改进，提高网络性能。Inception V3研究了Inception Module和Reduction Module的组合，通过多次卷积和非线性变化，极大的提升了网络性能。

- 参考实现：

  ```
  url=https://github.com/pytorch/examples.git
  commit_id=507493d7b5fab51d55af88c5df9eadceb144fb67
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.6.0 |
  | PyTorch 1.8 | torchvision==0.9.1 |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。

## 准备数据集

1. 获取数据集。

   下载 `ImageNet` 开源数据集，将数据集上传到服务器任意路径下并解压。

   数据集目录结构参考如下所示。

   ```
   ├── ImageNet
         ├──train
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...
              ├──...
         ├──val
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/ # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/ # 8卡性能
     ```

   - 多机多卡性能数据获取流程。
     ```
     1. 多机环境准备
     2. 开始训练，每个机器请按下面提示进行配置
     bash ./test/train_performance_multinodes.sh --data_path=数据集路径 --batch_size=单卡batch_size*单机卡数 --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
     ```
   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   -a                             // 模型名称
   --data                         // 数据集路径
   -j                             // 最大线程数
   --output_dir                   // 输出目录
   -b                             // 训练批次大小
   --lr                           // 初始学习率
   --print-freq                   // 打印频率
   --epochs                       // 重复训练次数
   --label-smoothing              // 标签平滑系数
   --wd                           // 权重衰减系数
   -p                             // 类别数量
   --amp                          // 使用混合精度
   --npu                          // 使用设备
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

## Inception_v3 training result

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V | - | 769.965 | 1      | O2       | 1.5 |
| 8p-竞品V | 79.634 | 5298.088 | 240      | O2       | 1.5 |
| 1p-NPU   | - | 811.51 | 1      | O2       | 1.8 |
| 8p-NPU   | 78.12 | 6487.75 | 240      | O2       | 1.8 |
# 版本说明

## 变更

2022.09.24：首次发布。

## FAQ

无。

# InceptionV3 模型推理指导

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

InceptionV3 模型是谷歌 Inception 系列里面的第三代模型，在 InceptionV2 模型的基础上，InceptionV3 通过分解卷积和新的正则化方法，极大地减少了计算开销。

+ 论文  
    [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)  
    Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna

+ 参考实现：  
    https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py

## 输入输出数据
+ 模型输入  
    | input-name | data-type | data-format |input-shape |
    | ---------- | --------- | ----------- | ---------- |
    | actual_input_1 | FLOAT32 | NCHW | bs x 3 x 299 x 299 | 

+ 模型输出  
    | output-name |  data-type | data-format |output-shape |
    | ----------- | ---------- | ----------- | ----------- |
    | output1      |  FLOAT32   | ND          | bs x 1000        |


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

## 安装

- 安装推理过程所需的依赖
    ```bash
    pip install -r requirements.txt
    ```

## 准备数据集

1. 获取原始数据集  
    本模型推理项目使用 ILSVRC2012 数据集验证模型精度，请在 [ImageNet官网](https://gitee.com/link?target=http%3A%2F%2Fimage-net.org%2F) 自行下载，并按照以下的目录结构存放图片与标签文件。   
    ```
    ├── imageNet/
        ├── val/
            ├──ILSVRC2012_val_00000001.JPEG
            ├──ILSVRC2012_val_00000002.JPEG
            ├──...
        ├── val_label.txt
    ```


2. 数据预处理  
    执行前处理脚本将原始数据转换为OM模型输入需要的bin/npy文件。
    ```bash
    python inceptionv3_preprocess.py --src_path /opt/npu/imageNet/val --save_path ./prep_dataset
    ```
    参数说明：
    + --src_path: 测试图片所在的目录路径
    + --save_path: 存放生成的bin文件的目录路径
    
    运行成功后，每张原始图片都会对应生成一个bin文件存放于 ./prep_dataset 目录下，总计50000个bin文件。


## 模型转换

1. PyTroch 模型转 ONNX 模型  
 
    使用在线推理的权重存放到当前目录，可参考命令：

    然后参考执行以下命令生成 ONNX 模型：
    ```bash
    python inceptionv3_pth2onnx.py --checkpoint ./checkpoint.pth --onnx ./inceptionv3.onnx
    ```
    参数说明：
    + --checkpoint: 预训练权重文件的路径。若不指定，则会通过在线方式获取。
    + --onnx: 生成ONNX模型的保存路径

2. ONNX 模型转 OM 模型  

    step1: 查看NPU芯片名称 \${chip_name}
    ```bash
    npu-smi info
    ```
    例如该设备芯片名为 910A，回显如下：
    ```
    +-------------------+-----------------+------------------------------------------------------+
    | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
    | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
    +===================+=================+======================================================+
    | 0       910A     | OK              | 15.8         42                0    / 0              |
    | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
    +===================+=================+======================================================+
    | 1       910A     | OK              | 15.4         43                0    / 0              |
    | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
    +===================+=================+======================================================+
    ```

    step2: ONNX 模型转 OM 模型
    ```bash
    # 配置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    chip_name=910A  # 根据 step1 的结果设值
    bs=128  # 根据需要自行设置 

    
    # 执行 ATC 进行模型转换
    atc --model=./inceptionv3.onnx \
        --framework=5 \
        --output=inceptionv3_bs${bs} \
        --input_format=NCHW \
        --input_shape="actual_input_1:${bs},3,299,299" \
        --log=error \
        --soc_version=Ascend${chip_name} \
        --insert_op_conf=inceptionv3_aipp.cfg
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
    安装ais_bench推理工具，请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。完成安装后，执行以下命令预处理后的数据进行推理。
    ```bash
    python -m ais_bench \
        --model inceptionv3_bs${bs}.om \
        --input ./prep_dataset/ \
        --output ./ \
        --output_dirname ./result/ \
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
    python -m ais_bench --model inceptionv3_bs${bs}.om --batchsize ${bs}
    ```
    执行完纯推理命令，程序会打印出与性能相关的指标，找到以关键字 **[INFO] throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。

3. 精度验证  

    执行后处理脚本，根据推理结果计算OM模型的精度：
    ```bash
    python inceptionv3_postprocess.py \
        --infer_results ./result/ \
        --anno_file /opt/npu/imageNet/val_label.txt \
        --metrics_json metrics.json
    ```
    参数说明：
    + --infer_results: 存放推理结果的目录路径
    + --anno_file: 标签文件路径
    + --metrics_json: 指定一个json文件用于保存指标信息。
    
    运行成功后，程序会将各top1~top5的正确率记录在 metrics.json 文件中，可执行以下命令查看：
    ```
    python -m json.tool metrics.json
    ```


----
# 性能&精度

在910A设备上，OM模型的精度为  **{Top1@Acc=77.31%, Top5@Acc=93.46%}**。

| 芯片型号   | BatchSize | 数据集      | 精度            | 
| --------- | --------- | ----------- | --------------- | 
|Ascend910A| 128         | ILSVRC2012  | Top1Acc=78.06% Top5@Acc=93.81% 

