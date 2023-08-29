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

TSM是一种通用且有效的时间偏移模块，它具有高效率和高性能，可以在达到3D CNN性能的同时，保持2D CNN的复杂性。TSM沿时间维度移动部分通道，从而促进相邻帧之间的信息交换。TSM可以插入到2D CNN中以实现零计算和零参数的时间建模。TSM可以扩展到在线设置，从而实现实时低延迟在线视频识别和视频对象检测。本模型是基于Sthv2数据库训练的TSM模型。

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                          | 数据排布格式 |
  | -------- | -------- | ----------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 48 x 3 x 256 x 256 | ND          |

- 输出数据

  | 输出数据  | 数据类型 | 大小    | 数据排布格式 |
  | -------- | -------- | ------- | ------------ |
  | output   | FLOAT32  | batchsize x 174 | ND           |

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

   本模型依赖decord，仅支持x86安装
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

   该模型使用Sthv2（Something-Something V2）的验证集进行测试
   ```shell
   mkdir -p mmaction2/data/sthv2/annotations
   ```
   将json文件下载解压缩后放在mmaction2/data/sthv2/annotations目录下，视频压缩包下载后放在mmaction2/data/sthv2目录下，并运行以下命令
   ```shell
   cd mmaction2/data/sthv2
   cat 20bn-something-something-v2-?? | tar zx
   mv 20bn-something-something-v2 videos
   cd ../../tools/data/sthv2
   bash extract_rgb_frames_opencv.sh
   bash generate_videos_filelist.sh
   bash generate_rawframes_filelist.sh
   cd ../../../../
   ```
   本项目默认将数据集存放于 `dataset=mmaction2/data/sthv2`

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行 `TSM_preprocess.py` 脚本，完成预处理。

   ```shell
   python3 TSM_preprocess.py \
          --data_root ${dataset}/rawframes/ \
          --ann_file ${dataset}/sthv2_val_list_rawframes.txt \
          --output_dir preprocess_bin
   ```
   - 参数说明：
     - `--batch_size`：预处理batch数。
     - `--data_root`：输入图片路径。
     - `--ann_file`： 输入图片对应的信息（由`generate_rawframes_filelist.sh`生成）：
     - `--output_dir`：输出结果路径
   运行成功后，在主目录下生成sthv2文件夹，包含preprocess_bin文件夹和sthv2.json文件。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      [TSM基于mmaction2预训练的权重文件](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_sthv2_rgb/tsm_r50_256h_1x1x8_50e_sthv2_rgb_20210816-032aa4da.pth)

   2. 导出onnx文件。
   
      1. 使用pytorch2onnx.py导出onnx文件。

         运行pytorch2onnx.py脚本。

         ```shell
         python3 mmaction2/tools/deployment/pytorch2onnx.py \
                mmaction2/configs/recognition/tsm/tsm_r50_1x1x8_50e_sthv2_rgb.py \
                ./tsm_r50_256h_1x1x8_50e_sthv2_rgb_20210816-032aa4da.pth \
                --output-file=./tsm_bs${bs}.onnx --softmax --shape ${bs} 48 3 256 256
         ```
         获得tsm_bs${bs}.onnx文件。
      
      2. 优化ONNX文件。(安装[auto-optimzer](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer)工具)

         ```shell
         python3 modify_onnx.py -m1 tsm_bs${bs}.onnx -m2 tsm_new_bs${bs}.onnx -bs ${bs}
         ```
         - 参数说明：
            - input_name(m1)：onnx文件路径。
            - output_name(m2): 优化后的onnx文件路径。
            - batch_size(bs)：batch size。

         获得tsm_new_bs${bs}.onnx文件

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
         atc --model=tsm_new_bs${bs}.onnx \
             --framework=5 \
             --output=tsm_bs${bs} \
             --input_format=ND \
             --log=error \
             --soc_version=Ascend${chip_name}

         ```

         - 参数说明：
            - model：为ONNX模型文件。
            - framework：5代表ONNX模型。
            - output：输出的OM模型。
            - input\_format：输入数据的格式。
            - log：日志级别。
            - soc\_version：处理器型号。

          运行成功后生成 `tsm_bs${bs}.om` 模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      ```shell
      python -m ais_bench \
          --model ./tsm_bs${bs}.om \
          --input ./sthv2/preprocess_bin \
          --output ./inference_result \
          --output_dirname out \
          --outfmt TXT
      ```
      推理后的输出默认在当前目录inference_result下。

   3. 精度验证。

      ```shell
      python TSM_postprocess.py  \
            --result_path=inference_result/out_summary.json \
            --info_path=sthv2/sthv2.info
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
         - model：om模型路径。
         - loop：循环次数。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>


  | NPU芯片型号 | Batch Size | 数据集      | 精度(wer)                | 性能 (fps)   |
  | ---------- | ---------- | ----------- | ------------------------ | ----------- |
  |  310P3     | 1          | Sthv2       | top1:0.6187 top5:0.8721  | 20.77       |
  |  310P3     | 4          | Sthv2       |                          | 18.41       |
  |  310P3     | 8          | Sthv2       |                          | 17.67       |
  |  310P3     | 16         | Sthv2       |                          | 17.12       |
  |  310P3     | 32         | Sthv2       |                          | 17.36       |