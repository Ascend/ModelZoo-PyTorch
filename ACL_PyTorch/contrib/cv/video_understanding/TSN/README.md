#  TSN 模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

------

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

TSN是一个经典的动作识别网络，在时间结构建模方面，采用稀疏时间采样策略，因为密集时间采样会有大量的冗余相似帧。然后提出可video-level的框架，在长视频序列中提取短片段，同时样本在时间维度均匀分布，由此采用segment结构来聚合采样片段的信息。

- 参考论文：

  [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859)

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmaction2
  branch=master
  commit_id=9ab8c2af52c561e5c789ccaf7b62f4b7679c103c
  model_name=TSN
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                           | 数据排布格式 |
  | -------- | -------- | ------------------------------ | ------------ |
  | input    | RGB_FP32 | batchsize x 75 x 3 x 256 x 256 | NTCHW        |

- 输出数据

  | 输出数据 | 数据类型 | 大小            | 数据排布格式 |
  | -------- | -------- | --------------- | ------------ |
  | output1  | FP32     | batchsize x 101 | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1** 版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fpies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.12.1  | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取本仓源码

2. 获取开源模型代码

   ```
   git clone https://github.com/open-mmlab/mmaction2.git
   cd mmaction2
   git checkout 9ab8c2af52c561e5c789ccaf7b62f4b7679c103c
   pip3 install -r requirements/build.txt
   pip3 install -v -e .
   cd ..
   ```
   
3. 安装必要依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   本模型使用UCF101数据集和对应annotations标签文件，详细介绍参见[UCF101官方下载地址](https://www.crcv.ucf.edu/data/UCF101.php)

   可使用开源代码仓的脚本下载和组织数据，具体步骤如下：
   
   1. 在已下载的源码包根目录下，执行如下命令：
   
      ```
      cd ./mmaction2/tools/data/ucf101
      bash download_annotations.sh  # 下载ucf101标签注释文件
      bash download_videos.sh       # 下载ucf101原始视频文件
      cd ../../../data/ucf101/
      unrar e ucf101.rar            # 默认下载的*.rar压缩包
      ```
   
      解压后得到的数据目录结构，仅供参考：
   
      ```
      ├──ucf101
          ├──videos
          ├──annotations
                ├── classInd.txt
                ├── testlist01.txt
                ├── testlist02.txt
                ├── testlist03.txt
                ├── trainlist01.txt
                ├── trainlist02.txt
                └── trainlist03.txt
                ...
      ```
   
   2. 在已下载的源码包根目录下，执行如下命令提取视频帧并生成对应的数据集文件列表：
   
      ```
      cd ./mmaction2/tools/data/ucf101    # 通过开源代码仓提供的数据处理脚本处理数据
      bash extract_rgb_frames_opencv.sh   # 通过opencv从视频中提取rgb帧
      bash generate_videos_filelist.sh    # 生成视频数据集list文件
      bash generate_rawframes_filelist.sh # 生成帧数据集list文件 
      ```
   
2. 数据预处理，将原始数据集转换为模型的输入数据。

   在已下载的源码包根目录下，执行tsn_ucf101_preprocess.py脚本，完成预处理。

   ```
   python3 tsn_ucf101_preprocess.py --data_root ./mmaction2/data/ucf101 --save_dir ${save_dir}
   ```

   参数说明：

   
     -   --data_root：原始数据根目录路径。
   
   
     -   --save_dir：输出的二进制文件（.bin）所在路径。
   
   预处理后会生成标签文件ucf101_1.info，并在指定文件夹生成二进制文件作为模型的输入
   
   预处理后的数据目录结构，仅供参考：
   
   ```
   -- ucf101
       |-- annotations
       |-- out_bin_1
       |-- rawframes
       |-- ucf101_1.info
       |-- ucf101_train_split_1_rawframes.txt
       |-- ucf101_train_split_1_videos.txt
       |-- ucf101_train_split_2_rawframes.txt
       |-- ucf101_train_split_2_videos.txt
       |-- ucf101_train_split_3_rawframes.txt
       |-- ucf101_train_split_3_videos.txt
       |-- ucf101_val_split_1_rawframes.txt
       |-- ucf101_val_split_1_videos.txt
       |-- ucf101_val_split_2_rawframes.txt
       |-- ucf101_val_split_2_videos.txt
       |-- ucf101_val_split_3_rawframes.txt
       |-- ucf101_val_split_3_videos.txt
       `-- videos
   ```

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      获取权重文件，并将其放入当前工作目录。

      ```
      wget -c https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb/tsn_r50_1x1x3_75e_ucf101_rgb_20201023-d85ab600.pth
      ```

   2. 导出onnx文件。

      在已下载的源码包根目录下，运行tsn_ucf101_pytorch2onnx.py脚本。

      ```
      python3 tsn_ucf101_pytorch2onnx.py mmaction2/configs/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb.py ./tsn_r50_1x1x3_75e_ucf101_rgb_20201023-d85ab600.pth --output-file=./tsn.onnx --verify
      ```

      参数说明：

      - --位置参数1：配置文件路径

      - --位置参数2：checkpoints文件路径

       - --output-file：转换后的.onnx文件输出路径
      
       - --verify：是否对照pytorch输出验证onnx模型输出

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

      2. 执行命令查看芯片名称（${chip_name}）。

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

   4. 执行ATC命令。

      ```
      # bs = [1, 4, 8, 16, 32]
      atc --framework=5 --model=tsn.onnx --output=tsn_bs${bs} --input_format=NCDHW --input_shape="image:${bs},75,3,256,256" --log=error --soc_version=Ascend${chip_name}
      ```

      运行成功后生成om模型文件。

      参数说明：

      - --model：为ONNX模型文件。
      - --framework：5代表ONNX模型。
      - --output：输出的OM模型。
      - --input_format：输入数据的格式。
      - --input_shape：输入数据的shape。
      - --log：日志级别。
      - --soc_version：处理器型号。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      ```
      python3 -m ais_bench --model=tsn_bs${bs}.om  --batchsize=${bs} \
      --input ${save_dir} --output result --output_dirname result_bs${bs} --outfmt TXT
      ```
      
      参数说明：
      
      - --model：om模型路径。
      - --batchsize：批次大小。
      - --input：输入数据所在路径。
      - --output：推理结果输出路径。
      - --output_dirname：推理结果输出子文件夹。
      - --outfmt：推理结果输出格式
   
3. 精度验证。

   调用tsn_ucf101_postprocess.py脚本与真值比对，可以获得精度数据。

   ```
   python3 tsn_ucf101_postprocess.py --result_path result/result_bs${bs} --data_root ./mmaction2/data/ucf101
   ```

   参数说明：

   - --result_path：推理结果所在路径。
   - --info_path：数据集标签文件。执行“tsn_ucf101_preprocess.py”脚本时生成的。


4. 可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

   ```
   python3 -m ais_bench --model=tsn_bs${bs}.om --loop=50 --batchsize=${bs}
   ```

   参数说明：

   - --model：om模型路径。
   - --batchsize：批次大小。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，TSN模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集 | 开源精度（Acc@1）                                            | 参考精度（Acc@1） |
| ----------- | ---------- | ------ | ------------------------------------------------------------ | ----------------- |
| Ascend310P3 | 1          | UCF101 | [83.03%](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsn/README.md#ucf-101) | 82.83%            |

| 芯片型号    | Batch Size | 参考性能（FPS） |
| ----------- | ---------- | --------------- |
| Ascend310P3 | 1          | 15.27           |
| Ascend310P3 | 4          | 21.36           |
| Ascend310P3 | 8          | 22.08           |
| Ascend310P3 | 16         | 22.17           |
| Ascend310P3 | 32         | 22.19           |