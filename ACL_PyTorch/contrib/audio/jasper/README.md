# Jasper模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Jasper是应用于自动语音识别（ASR）的端到端声学模型，该模型在不借助任何外部数据的情况下在LibriSpeech数据集上取得了SOTA的结果

- 参考实现：

  ```
  url=https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper
  commit_id=15af494a8e7e0c33fcbdc6ef9cc12e3929e313aa
  code_path= ACL_PyTorch/contrib/audio/jasper
  model_name=jasper
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小               | 数据排布格式 |
  | -------- | -------- | ------------------- | ------------ |
  | input    | FP16     | batchsize x 64 x 4000 | ND           |

- 输出数据

  | 输出数据 | 数据类型 | 大小                | 数据排布格式 |
  | -------- | -------- | ------------------- | ------------ |
  | output   | FP16     | batchsize x 2000 x 29 | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

**表 1**  版本配套表

| 配套                                                            | 版本    | 环境准备指导                                                                                          |
| --------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
| 固件与驱动                                                      | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                            | 6.0.RC1 | -                                                                                                     |
| Python                                                          | 3.7.5   | -                                                                                                     |
| PyTorch                                                         | 1.11.0   | -                                                                                                     |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```shell
   git clone https://github.com/NVIDIA/DeepLearningExamples.git
   cd DeepLearningExamples
   git reset --hard 15af494a8e7e0c33fcbdc6ef9cc12e3929e313aa
   mv ../Jasper.patch ./
   git apply Jasper.patch
   cd PyTorch/SpeechRecognition/Jasper      # 将本仓库中代码拷贝到此目录下执行
   ```
   将requirements.txt，Jasper_preprocess.py，Jasper_pth2onnx.py，Jasper_postprocess.py拷贝到DeepLearningExamples/PyTorch/SpeechRecognition/Jasper目录下。

2. 安装依赖。

   ```shell
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持 LibriSpeech 语音包的验证集。请用户自行获取 [LibriSpeech-test-other.tar.gz](https://www.openslr.org/resources/12/test-other.tar.gz)数据集，然后上传数据集到推理服务器任意目录并解压（如：`dataset_dir=/home/HwHiAiUser/dataset`）。

   目录结构如下：

   ```shell
   ├──LibriSpeech
       ├──test-other
         ├──8461
         ├──8280
         ......
       ├──SPEAERS.TXT
       ├──README.TXT
       ├──LICENSE.TXT
       ├──CHAPTERS.TXT
       ├──BOOKS.TXT
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   1. 使用代码仓 Japser/utils目录下的convert_librispeech.py 脚本原始数据（.flac）转化为语音文件（.wav）。

      ```shell
      python3 ./utils/convert_librispeech.py \
            --input_dir ${dataset_dir}/test-other \
            --dest_dir ${dataset_dir}/test-other-wav \
            --output_json ${dataset_dir}/librispeech-test-other-wav.json
      ```

      - 参数说明：
         - input_dir： 原始数据验证集（.flac）所在路径
         - dest_dir： 输出文件（.wav）所在路径
         - output_json：wav对应的元信息json文件

      每个 .flac 对应生成一个.wav文件，保存在${dataset_dir}/test-other-wav目录下。
      
   2. 执行Jasper_preprocess.py脚本，完成预处理。
      ```shell
      python3 Jasper_preprocess.py --dataset_dir ${dataset_dir} \
              --val_manifests ${dataset_dir}/librispeech-test-other-wav.json \
              --model_config configs/jasper10x5dr_speedp-online_speca.yaml \
              --max_duration 40 \
              --pad_to_max_duration \
              --save_bin_0 ./prep_data_0 \
              --save_bin_1 ./prep_data_1 \
              --json_file ./agg_txts.json
      ```
      - 参数说明：
         - dataset_dir：数据集所在路径。
         - val_manifests：数据信息的json文件。
         - model_config：模型配置文件。
         - max_duration：最大pad数目。
         - pad_to_max_duration：是否pad。
         - save_bin_0：input0存储路径。
         - save_bin_1：input1存储路径。
         - json_file：存储处理后的数据信息的json文件

      运行成功生成prep_data_0，prep_data_1文件夹和agg_txts.json文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       - 点击[Link](https://ngc.nvidia.com/catalog/models/nvidia:jasper_pyt_ckpt_amp)获取官方发布的Jasper权重文件nvidia_jasper_210205.pt。

         ```
         mkdir checkpoint
         ```

         将nvidia_jasper_210205.pt 文件移动到checkpoint文件夹下。

   2. 导出onnx文件。

      运行Jasper_pth2onnx.py脚本，导出onnx模型。

      ```
      python3 Jasper_pth2onnx.py  checkpoint/nvidia_jasper_210205.pt jasper.onnx
      ```

      该转换过程执行时间较长请耐心等待。运行成功后在当前目录生成 jasper.onnx 模型文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```shell
         source /usr/local/Ascend/ascend-tooklit/set_env.sh
         ```

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```shell
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
         atc --model=jasper.onnx \
             --framework=5 \
             --input_format=ND \
             --input_shape="feats:${bs},64,4000;feat_lens:${bs}" \
             --output=jasper_bs${bs} \
             --soc_version=Ascend${chip_name} \
             --log=error
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

         运行成功后生成jasper_bs${bs}.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

      ```shell
      python3 -m ais_bench --model jasper_bs${bs}.om --input "prep_data_0,prep_data_1" --output ./ --output_dirname result
      ```
      -  参数说明：
         - model：om文件路径。
         - input：模型输入文件路径。
         - output：输出文件路径。
         - output_dirname：输出文件子目录。
      运行成功后，得到result文件夹。

   3. 精度验证。

      调用脚本与数据集标签比对，可以获得wer数据。

      ```shell
      python3 Jasper_postprocess.py \
            --model_config configs/jasper10x5dr_speedp-online_speca.yaml \
            --save_bin ./result \
            --json_file ./agg_txts.json
      ```

      - 参数说明：

        - model_config：模型配置文件。
        - save_bin：推理结果存储路径。
        - json_file：预处理得到的json文件。

   4. 性能验证

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：执行推理。

      ```shell
      python3 -m ais_bench --model jasper_bs${bs}.om --loop 20
      ```

      - 参数说明：

         - model：om模型。
         - loop：纯推理循环次数。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集                  | 精度  | 性能  |
| -------- | ---------- | ---------------------- | ----- | ----- |
| 310P3    | 1          | LibriSpeech-test-other | 9.709 | 41.37 |
| 310P3    | 4          | LibriSpeech-test-other |       | 37.64 |
| 310P3    | 8          | LibriSpeech-test-other |       | 36.57 |
| 310P3    | 16         | LibriSpeech-test-other |       | 34.39 |
| 310P3    | 32         | LibriSpeech-test-other |       | 34.38 |
| 310P3    | 64         | LibriSpeech-test-other |       | 34.46 |