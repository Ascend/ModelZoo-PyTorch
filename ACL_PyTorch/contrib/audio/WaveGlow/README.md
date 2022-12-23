# WaveGlow模型-推理指导


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

WaveGlow是一种flow-based网络，能够从mel频谱图生成高质量的语言。WaveGlow结合了Glow和WaveNet，提供了快速、高效以及高质量的音频合成，并且不依赖自动回归。


- 参考实现：

  ```
  url=https://github.com/NVIDIA/waveglow.git
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FLOAT32 | batchsize x 80 x mel_seq | ND         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x audio_seq | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动 

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.6.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/NVIDIA/waveglow.git
   cd waveglow
   git submodule init
   git submodule update
   git apply ../WaveGlow.patch
   cd ..
   export PYTHONPATH=./waveglow:./waveglow/tacotron2:${PYTHONPATH}
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   下载[LJSpeech-1.1数据集](https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2)，解压至当前目录

   ```
   wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
   tar jxvf LJSpeech-1.1.tar.bz2
   ```

   目录结构如下：

   ```
   ${data_path}
      |-- LJSpeech-1.1
         |-- wavs
         |    |-- LJ001-0001.wav
         |    |-- LJ001-0002.wav
         |    |-- …
         |    |-- LJ050-0278
         |-- metadata.csv
      |-- README
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行WaveGlow_preprocess.py脚本，完成预处理。

   ```
   # 测试集为LJSpeech-1.1数据集中前10条数据
   ls LJSpeech-1.1/wavs/*.wav | head -n10 > test_files.txt
   mkdir data
   python3 WaveGlow_preprocess.py -f ./test_files.txt -c waveglow/config.json -o ./data/
   ```

   - 参数说明：

     -   -f 测试集数据名。
     -   -c 模型配置json文件。
     -   -o 前处理结果存放路径。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      [waveglow_256channels_universal_v5.pt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/WaveGlow/PTH/waveglow_256channels_universal_v5.pt)

   2. 导出onnx文件。

      1. 使用WaveGlow_pth2onnx.py导出onnx文件。

         ```
         python3 WaveGlow_pth2onnx.py -i ./waveglow_256channels_universal_v5.pt -o ./
         ```

         - 参数说明：

           -   -f 测试集数据名。
           -   -c 模型配置json文件。
           -   -o 前处理结果存放路径。

         获得waveglow.onnx文件。

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

         ```
         atc --model=waveglow.onnx \
               --output=waveglow \
               --input_shape="mel:1,80,-1" \
               --framework=5 \
               --input_format=ND \
               --soc_version=Ascend${chip_name} \
               --log=error \
               --dynamic_dims="154;164;443;490;651;699;723;760;832;833" 
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --dynamic_dims：动态模型dim档位

           运行成功后生成<u>***waveglow.om***</u>模型文件。

2. 开始推理验证。

   1. 使用ais_bench工具进行推理。

      ais_bench工具获取及使用方式请点击查看[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

        ```
        python3 -m ais_bench --model=./waveglow.om --input=data --output=./result --outfmt=BIN --auto_set_dymdims_mode=1
        ```

        -   参数说明：

            -   --model：om文件路径
            -   --input：预处理后二进制目录。
            -   --output：推理结果输出路径。
            -   --outfmt：推理结果输出格式。

        推理后的输出默认在当前目录result下。

        >**说明：** 
        >执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见。

   3. 精度验证。

      调用脚本与数据集标签比对，可以获得Accuracy数据。

      ```
      mkdir synwavs
      python3 WaveGlow_postprocess.py -f ./result/${output_dir} -o ./synwavs/
      ```

      - 参数说明：

        - -f：推理结果输出路径。
        - -o：后处理结果存放路径。

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3 -m ais_bench --model=${om_model_path} --loop=20 --dymDims mel:1,80,${mel_seq} --batchsize=1
        ```

      - 参数说明：
        - --model：om模型路径
        - --batchsize：batch大小



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

|    芯片型号     |     Batch Size    |      mel_seq      |       性能      |
| -------------- | ----------------- | ----------------- | --------------- |
|     310P3      |        1          |        154        |      3.086      |
|     310P3      |        1          |        164        |      2.893      |
|     310P3      |        1          |        443        |      1.034      |
|     310P3      |        1          |        490        |      0.935      |
|     310P3      |        1          |        651        |      0.699      |
|     310P3      |        1          |        699        |      0.648      |
|     310P3      |        1          |        723        |      0.627      |
|     310P3      |        1          |        760        |      0.596      |
|     310P3      |        1          |        832        |      0.546      |
|     310P3      |        1          |        833        |      0.545      |