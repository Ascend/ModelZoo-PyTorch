# TSM模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

TSM是一种通用且有效的时间偏移模块，它具有高效率和高性能，可以在达到3D CNN性能的同时，保持2D CNN的复杂性。TSM沿时间维度移动部分通道，从而促进相邻帧之间的信息交换。TSM可以插入到2D CNN中以实现零计算和零参数的时间建模。TSM可以扩展到在线设置，从而实现实时低延迟在线视频识别和视频对象检测。

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                          | 数据排布格式 |
  | -------- | -------- | ----------------------------- | ------------ |
  | pos      | RGB_FP32 | batchsize x 8 x 3 x 224 x 224 | NCDHW        |

- 输出数据

  | 输出数据 | 数据类型 | 大小    | 数据排布格式 |
  | -------- | -------- | ------- | ------------ |
  | output_0 | FLOAT32  | 1 x 101 | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                            | 版本    | 环境准备指导                                                                                          |
  | --------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
  | 固件与驱动                                                      | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 5.1.RC2 | -                                                                                                     |
  | Python                                                          | 3.7.5   | -                                                                                                     |
  | PyTorch                                                         | 1.12.0  | -                                                                                                     |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码

    ```shell
    git clone https://github.com/open-mmlab/mmaction2.git
    cd mmaction2
    git reset --hard 5fa8faa
    cd ..
    ```

2. 安装依赖

    ```shell
    pip install -r requirements.txt
    ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

    该模型使用 [UCF-101](https://www.crcv.ucf.edu/research/data-sets/ucf101/) 的验证集进行测试，数据集下载步骤如下
    ```shell
    cd ./mmaction2/tools/data/ucf101
    bash download_annotations.sh
    bash download_videos.sh
    bash extract_rgb_frames_opencv.sh
    bash generate_videos_filelist.sh
    bash generate_rawframes_filelist.sh
    ```
    本项目默认将数据集存放于 `dataset=mmaction2/data/ucf101`

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行 `TSM_preprocess.py` 脚本，完成预处理。

   ```shell
   python TSM_preprocess.py \
          --batch_size 1 \
          --data_root ${dataset}/rawframes/ \
          --ann_file ${dataset}/ucf101_val_split_1_rawframes.txt \
          --output_dir preprocess_bin
   ```
   - 参数说明：
     - `--batch_size`：预处理batch数。
     - `--data_root`：输入图片路径。
     - `--ann_file`： 输入图片对应的信息（由`generate_rawframes_filelist.sh`生成）：
     - `--output_dir`：输出结果路径

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      [TSM基于mmaction2预训练的权重文件](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb_20210630-1fae312b.pth)

   2. 导出onnx文件。

      ```shell
        # 本文以batchsize=1为例进行说明
        python mmaction2/tools/deployment/pytorch2onnx.py \
                mmaction2/configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb.py \
                ./tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb_20210630-1fae312b.pth \
                --output-file=tsm.onnx --softmax --shape 1 8 3 224 224
      ```
      建议使用onnxsim简化onnx模型
      ```shell
      onnxsim tsm.onnx tsm_sim.onnx
      ```

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

          ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
          ```

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

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

          ```shell
          bs=1    # 以batchsize=1为例，其它batch自行修改
          atc --model=tsm.onnx \
              --framework=5 \
              --output=tsm_bs1 \
              --input_format=NCDHW \
              --log=error \
              --soc_version=${chip_name}

          ```

            - 参数说明：
              -   --model：为ONNX模型文件。
              -   --framework：5代表ONNX模型。
              -   --output：输出的OM模型。
              -   --input\_format：输入数据的格式。
              -   --input\_shape：输入数据的shape。
              -   --log：日志级别。
              -   --soc\_version：处理器型号。

              运行成功后生成 `tsm_bs1.om` 模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请点击本链接进行安装ais_bench推理工具，以及查看具体使用方法(https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)  

   2. 执行推理。

      ```shell
      bs=1
      python -m ais_bench \
          --model ./tsm_bs1.om \
          --input ./ucf101/preprocess_bin \
          --output ./inference_result \
          --output_dirname bs$bs \
          --outfmt TXT \
          --batchsize $bs

      ```
      >**说明：**
      >执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见。

   3. 精度验证。

      ```python
      python TSM_postprocess.py  \
                --result_path=inference_result/bs1_summary.json \
                --info_path=ucf101/ucf101.info \
      ```
      - 参数说明：
        - --result_path：推理结果对应的文件夹
        - --info_path：数据集info文件路径

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

- 精度

  | Batch_size | Framework | Container | Precision | Dataset | Accuracy                | Ascend AI Processor |
  | ---------- | --------- | --------- | --------- | ------- | ----------------------- | ------------------- |
  | 1          | PyTorch   | NA        | fp16      | UCF101  | top1:0.9402 top5:0.9958 | Ascend 310          |
  | 1          | PyTorch   | NA        | fp16      | UCF101  | top1:0.9402 top5:0.9958 | Ascend 310P         |

- 性能

  | Model | Batch Size | 310 (FPS/Card) | 310p (FPS/Card) | T4 (FPS/Card) | 310p/310 | 310p/T4 |
  | ----- | ---------- | -------------- | --------------- | ------------- | -------- | ------- |
  | TSM   | 1          | 24.80          | 171.04          | 98.01         | 7.16     | 1.81    |
  | TSM   | 4          | 22.48          | 132.23          | 107.90        | 5.88     | 1.22    |
  | TSM   | 8          | 20.25          | 123.814         | 100.0         | 6.11     | 1.23    |
  | TSM   | 16         | 19.86          | 119.71          | 101.89        | 6.02     | 1.17    |
  | TSM   | 32         | 18.90          | 99.78           | 100.91        | 5.27     | 0.98    |
