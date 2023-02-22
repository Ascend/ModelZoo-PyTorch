# DPT模型-推理指导


- [概述](#概述)
- [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
- [模型推理性能与精度](#模型推理性能与精度)

## 概述

Dense Prediction Transformer (DPT) 是一种基于 transformer 以 encoder-decoder 为主要结构的神经网络，用于图像的密集预测。模型使用 ViT 作为 encoder 结构，使用卷积层作为 decoder 结构，可以接受不同大小的输入。


- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmsegmentation/tree/master/configs/dpt
  commit_id=b51670b61339e5b10c5ab6c277de6b6a387fdff0
  code_path=mmsegmentation/mmseg/models/decode_heads/dpt_head.py
  model_name=DPT
  ```

### 输入输出数据

- 输入数据

  | 输入数据 | 数据类型 | 大小 | 数据排布格式 |
  | -------- | -------- | ------------ | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 512 x 512 | NCHW |


- 输出数据

  | 输出数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output   | FLOAT32  | batchsize x 150 x 512 x 512 | NCHW |

## 推理环境准备

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.11.0  | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以 CANN 版本选择实际固件与驱动版本。 | \       | \                                                            |

## 快速上手

### 获取源码

1. 获取源码。**<u>*此处获取指获取第三方开源代码仓的命令*</u>**

   ```bash
   git clone -b v0.28.0 https://github.com/open-mmlab/mmsegmentation.git
   cd mmsegmentation
   git reset --hard b51670b61339e5b10c5ab6c277de6b6a387fdff0
   cd ..
   ```

2. 安装依赖。

   ```bash
   pip3 install -r requirements.txt
   cd mmsegmentation
   pip3 install -v -e .
   cd ..
   mim install mmcv-full==1.6.0
   mkdir dpt_work
   ```

### 准备数据集

1. 获取原始数据集。（解压命令参考 `tar –xvf  \*.tar` 与 `unzip \*.zip`）

   本推理项目使用 ADE20K 的 2000 张验证集图片来验证模型精度，请进入 [ADE20K官网](http://groups.csail.mit.edu/vision/datasets/ADE20K/) 自行下载数据集（需要先注册）。下载后请自行解压或参考以下命令：
   
    ```shell
    mkdir -p data/ade
    unzip ${zip_path}/ADEChallengeData2016.zip -d data/ade/
    ```
	
    `${zip_path}` 为 ADEChallengeData2016.zip 的路径。最终，验证集原始图片与标注图片的存放结构如下：
	
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

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行前处理脚本将原始数据集中的 .jpg 图片转换为 OM 模型输入需要的 .bin 文件。
   
    ```bash
    python3 dpt_preprocess.py \
        --config mmsegmentation/configs/dpt/dpt_vit-b16_512x512_160k_ade20k.py \
        --save-dir dpt_work/val_bin/
    ```
	
    参数说明：
    + --config : 模型配置文件路径
    + --save-dir : 存放生成的 bin 文件的目录路径
   
    原始图片在预处理的时候会进行滑窗操作，一张图片对应一个或多个滑窗，每个滑窗单独保存成一个 .bin 文件。预处理脚本运行结束后，2000 张原始图会生成 3686 个 .bin 文件，存放于 `dpt_work/val_bin` 目录中。

### 模型推理

1. 模型转换。

   使用 PyTorch 将 .pth 模型权重文件转换为 .onnx 文件，再使用 ATC 工具将 .onnx 文件转为离线推理模型文件 .om 文件。

   1. 获取权重文件。

       本推理项目使用开源仓提供的预训练好的 [权重文件](https://download.openmmlab.com/mmsegmentation/v0.5/dpt/dpt_vit-b16_512x512_160k_ade20k/dpt_vit-b16_512x512_160k_ade20k-db31cf52.pth)，下载完成后将权重 .pth 文件存放于 dpt_work 目录下。
	
    ```shell
    wget \
        https://download.openmmlab.com/mmsegmentation/v0.5/dpt/dpt_vit-b16_512x512_160k_ade20k/dpt_vit-b16_512x512_160k_ade20k-db31cf52.pth \
        -P dpt_work
    ```


   2. 导出 onnx 文件。

      1. 使用 `dpt_pth2onnx.py` 导出 onnx 文件。
         
         执行命令：
      	
         ```shell
         python3 dpt_pth2onnx.py \
             --config ./mmsegmentation/configs/dpt/dpt_vit-b16_512x512_160k_ade20k.py \
             --checkpoint ./dpt_work/dpt_vit-b16_512x512_160k_ade20k-db31cf52.pth \
             --onnx ./dpt_work/dpt_bs${bs}.onnx \
             --batchsize ${bs}
         ```

        参数说明：
        + --config : 模型配置文件路径
        + --checkpoint : 预训练权重所在路径
        + --onnx : 生成 ONNX 模型的保存路径
        + --batchsize : 模型输入的 batchsize ，默认为 1

        命令中的 `${bs}` 表示模型输入的 batchsize ，比如将 `${bs}` 设为 1 ，运行结束后，在 dpt_work 目录下会生成 dpt_bs1.onnx 模型。
        `${bs}` 可支持的值为：1，4，8。

      2. 优化 ONNX 文件。

         使用 onnx-simplifier 工具简化 onnx 模型，命令如下：

         ```bash
         onnxsim ./dpt_work/dpt_bs${bs}.onnx ./dpt_work/dpt_sim_bs${bs}.onnx 
         ```

         在 dpt_work 路径中获得`dpt_sim_bs${bs}.onnx`文件。

   3. 使用 ATC 工具将 ONNX 模型转 OM 模型。

      1. 配置环境变量。

         ```bash
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

      2. 执行命令查看芯片名称（`${chip_name}`）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         回显如下：
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

      3. 执行ATC命令。

         ```bash
         atc --framework=5 \
             --model=./dpt_work/dpt_sim_bs${bs}.onnx \
             --input_shape="input:${bs},3,512,512" \
             --output=./dpt_work/dpt_bs${bs} \
             --input_format=NCHW \
             --log=error \
             --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为 ONNX 模型文件。
           -   --framework：5 代表 ONNX 模型。
           -   --output：输出的 OM 模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的 shape 。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           
           运行成功后在 dpt_work 目录下生成 `dpt_bs${bs}.om` 模型文件。

2. 开始推理验证。

   1. 使用 ais-infer 工具进行推理。

      ais-infer 工具获取及使用方式请查看 [ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)。

   2. 执行推理。

        ```bash
        python3 -m ais_bench \
             --model=./dpt_work/dpt_bs${bs}.om \
             --input=./dpt_work/val_bin/ \
             --output=./dpt_work/ \
             --batchsize=${bs}
        ```

        -   参数说明：
             -   --model : om 模型。
             -   --input : 输入数据路径。
             -   --output : 推理结果保存路径。
             -   --batchsize : om 模型的 batchsize 。
   
        推理完成后在当前 dpt_work 路径下生成推理结果保存目录，其命名格式为 xxxx_xx_xx-xx_xx_xx(年\_月\_日-时\_分\_秒)，如 2022_08_18-06_55_19。
   
        >**说明：** 
        >执行 ais-infer 工具请选择与运行环境架构相同的命令。参数详情请参见 [ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)。
   
   3. 精度验证。
   
      调用后处理脚本 `dpt_postprocess.py` ,将推理结果与标签进行对比，获得mIoU精度数据。
   
      ```bash
      python3 dpt_postprocess.py \
        --config mmsegmentation/configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py \
        --infer-results ./dpt_work/${xxxx_xx_xx-xx_xx_xx}
      ```
   
      -   参数说明：
   
             -   --config : 数据处理配置文件。
             -   --infer-results : om 模型推理结果路径，`${xxxx_xx_xx-xx_xx_xx}` 表示 ais_infer 生成的推理结果目录。
   
   4. 性能验证。
   
      可使用 ais_infer 推理工具的纯推理模式验证不同 batch_size 的 om 模型的性能，参考命令如下：
   
      ```bash
      python3 -m ais_bench --model=${om_model_path} --loop=50 --batchsize=${batch_size}
      ```
   
      - 参数说明：
        - --model : om模型。
        - --loop : 推理次数。
        - --batchsize : om 模型的 batchsize 。

## 模型推理性能与精度

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| ---------- | --------- | ------ | ---------- | -------|
| Ascend310P | 1         | ADE20K | mIoU=48.37 | 24.120 |
| Ascend310P | 4         | ADE20K | mIoU=48.37 | 19.952 |
| Ascend310P | 8         | ADE20K | mIoU=48.37 | 21.628 |

注：由于内存限制，该模型支持的 batchsize 为 1, 4, 8 。性能最优 batchsize 为 1 。