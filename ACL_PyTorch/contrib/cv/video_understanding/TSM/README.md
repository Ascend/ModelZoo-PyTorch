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
  | input    | RGB_FP32 | batchsize x 8 x 3 x 224 x 224 | NCDHW        |

- 输出数据

  | 输出数据  | 数据类型 | 大小    | 数据排布格式 |
  | -------- | -------- | ------- | ------------ |
  | output   | FLOAT32  | batchsize x 101 | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                            | 版本    | 环境准备指导                                                                                          |
  | --------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
  | 固件与驱动                                                      | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 6.0.RC1 | -                                                                                                     |
  | Python                                                          | 3.7.5   | -                                                                                                     |
  | PyTorch                                                         | 1.9.0  | -                                                                                                     |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码

    ```shell
    git clone https://github.com/open-mmlab/mmaction2.git
    cd mmaction2
    git reset --hard 5fa8faa
    pip3 install -e .
    cd ..
    ```

2. 安装依赖

    ```shell
    pip3 install -r requirements.txt
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
    cd ../../../../
    ```
    本项目默认将数据集存放于 `dataset=mmaction2/data/ucf101`

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行 `TSM_preprocess.py` 脚本，完成预处理。

   ```shell
   python3 TSM_preprocess.py \
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
   
      1. 使用pytorch2onnx.py导出onnx文件。

         运行pytorch2onnx.py脚本。

         ```shell
         python3 mmaction2/tools/deployment/pytorch2onnx.py \
                mmaction2/configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb.py \
                ./tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb_20210630-1fae312b.pth \
                --output-file=./tsm_bs${bs}.onnx --softmax --shape ${bs} 8 3 224 224
         ```
         获得tsm_bs${bs}.onnx文件。
      
      2. 优化ONNX文件。(安装[auto-optimzer](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer)工具)

         ```shell
         onnxsim tsm_bs${bs}.onnx tsm_sim_bs${bs}.onnx
         python3 modify_onnx.py -m1 tsm_sim_bs${bs}.onnx -m2 tsm_sim_new_bs${bs}.onnx
         ```
         - 参数说明：
            - --input_name(m1)：onnx文件路径。
            - --output_name(m2): 优化后的onnx文件路径。

         获得tsm_sim_new_bs${bs}.onnx文件

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
          atc --model=tsm_sim_new_bs${bs}.onnx \
              --framework=5 \
              --output=tsm_bs${bs} \
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

          运行成功后生成 `tsm_bs${bs}.om` 模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      ```shell
      python -m ais_bench \
          --model ./tsm_bs${bs}.om \
          --input ./ucf101/preprocess_bin \
          --output ./inference_result \
          --output_dirname out \
          --outfmt TXT
      ```
      推理后的输出默认在当前目录inference_result下。

   3. 精度验证。

      ```shell
      python TSM_postprocess.py  \
            --result_path=inference_result/out_summary.json \
            --info_path=ucf101/ucf101.info
      ```
      - 参数说明：
        - --result_path：推理结果对应的文件夹
        - --info_path：数据集info文件路径

   4. 性能验证。
      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
      ```shell
      python3 -m ais_bench --model=tsm_bs${bs}.om --loop=20
      ```
      - 参数说明：
      - --model：om模型路径。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>


| Batch Size | 数据集      | 精度(wer)                | 310P3 | 310B1 |
| ---------- | ----------- | ------------------------ | ----------- | ---------- |
| 1          | UCF-101     | top1:0.9448 top5:0.9963  | 194.07      | 27.05 |
| 4          | UCF-101     |   | 161.49      | 26.82 |
| 8          | UCF-101     |   | 157.07      | 24.86 |
| 16         | UCF-101     |   | 156.06      | 16.38 |
| 32         | UCF-101     |   | 143.98      | 16.34 |
| 64         | UCF-101     |   | 134.72      | 16.48 |
|  |  | **最优性能** | **194.07** | **27.05** |