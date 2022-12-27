# FastPitch模型-推理指导


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

Fastpitch模型由双向 Transformer 主干（也称为 Transformer 编码器）、音调预测器和持续时间预测器组成。 在通过第一组 N 个 Transformer 块、编码后，信号用基音信息增强并离散上采样。 然后它通过另一组 N个 Transformer 块，目的是平滑上采样信号，并构建梅尔谱图。


- 参考实现：

  ```
  url=https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 200 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 80 x 900 | ND           |


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
   git clone https://github.com/NVIDIA/DeepLearningExamples
   git clone https://github.com/NVIDIA/dllogger.git
   cd dllogger
   git checkout 26a0f8f1958de2c0c460925ff6102a4d2486d6cc
   cd ..
   export PYTHONPATH=dllogger:${PYTHONPATH}
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   ```
   wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
   tar -xvjf LJSpeech-1.1.tar.bz2
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   生成输入数据，并准备输出标签和pth权重的输出数据。本模型的验证集大小为100，具体信息在phrases/tui_val100.tsv文件中。

   - FastPitch模型的输入数据是由文字编码组成，输入长度不等，模型已经将其补零成固定长度200。将输入数据转换为bin文件方便后续推理，存入test/input_bin文件夹下，且生成生成数据集预处理后的bin文件以及相应的info文件。
   - 在语音合成推理过程中，输出为mel图谱，本模型的输出维度为batch_size×900×80。将其输出tensor存为pth文件存入test/mel_tgt_pth文件夹下。
   - 同时，为了后面推理结束后将推理精度与原模型pth权重精度进行对比，将输入数据在pth模型中前传得到的输出tensor存为pth文件存入test/mel_out_pth文件夹下。

   以上步骤均执行下面指令完成：
   
   ```
   python3 DeepLearningExamples/PyTorch/SpeechSynthesis/FastPitch/prepare_dataset.py --wav-text-filelists DeepLearningExamples/PyTorch/SpeechSynthesis/FastPitch/filelists/ljs_audio_text_val.txt --n-workers 16 --batch-size 1 --dataset-path ./LJSpeech-1.1 --extract-mels --f0-method pyin
   ```

   ```
   python3 data_process.py -i phrases/tui_val100.tsv -o ./output/audio_tui_val100 --log-file ./output/audio_tui_val100/nvlog_infer.json --fastpitch pretrained_models/fastpitch/nvidia_fastpitch_210824.pt --waveglow pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt
   ```


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      ```
      wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/FastPitch/PTH/nvidia_fastpitch_210824.pt
      wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/FastPitch/PTH/nvidia_waveglow256pyt_fp16.pt
      ```
      （waveglow为语音生成器，不在本模型范围内, 但为了确保代码能正常运行，需要下载）

   2. 导出onnx文件。

      1. 使用pth2onnx.py导出onnx文件。

         ```
         python3 pth2onnx.py -i phrases/tui_val100.tsv -o ./output/audio_tui_val100 --log-file ./output/audio_tui_val100/nvlog_infer.json --fastpitch nvidia_fastpitch_210824.pt --waveglow nvidia_waveglow256pyt_fp16.pt --wn-channels 256 --energy-conditioning --batch-size 1
         ```

         获得FastPitch.onnx文件。

      2. 优化ONNX文件。

         ```
         python3 -m onnxsim FastPitch.onnx FastPitch_sim.onnx --dynamic-input-shape --input-shape 1,200
         ```

         获得FastPitch_sim.onnx文件。

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
         atc --framework=5 --model=FastPitch_sim.onnx --output=FastPitch_bs1 --input_format=ND --input_shape="input:1,200" --out_nodes='Transpose_2044:0' --log=debug --soc_version=Ascend${chip_name} 
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>***FastPitch_bs1.om***</u>模型文件。

2. 开始推理验证。

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 配置环境变量。

      ```
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      ```

   3. 执行推理。

        ```
        python3 -m ais_bench --model FastPitch_bs1.om --input test/input_bin --output result --output_dirname output_bs1 --outfmt BIN
        ```

        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入数据路径。
             -   --output：推理结果输出路径。
             -   --outfmt：推理结果输出格式。

        推理后的输出默认在当前目录result下。

        >**说明：** 
        >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

   4. 精度验证。

      调用脚本与数据集标签比对，可以获得Accuracy数据。

      ```
      python3 infer_test.py result/output_bs1
      ```

   5. 性能验证。

      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om文件路径
        - --batchsize：batch大小



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

|mel_loss  |   om     |     pth   |
| -------- | -------- | --------- |
|bs1	     |  11.246  |   11.265  |
|bs16	     |  11.330  |   11.265  |


| Model     | Batch Size |T4 Throughput/Card |  310 Throughput/Card |  310P3 Throughput/Card |
| --------- | ---------- | ------------------ | -------------------- | --------------------- |
| FasfPitch | 1          | 28.828             | 54.1476              |         90.2718       |
| FasfPitch | 4          | -                  | 51.728               |         123.7534      |
| FasfPitch | 8          | -                  | 51.3684              |         126.9909      |
| FasfPitch | 16         | 64.94              | 51.714               |         124.2424      |
| FasfPitch | 32         | -                  | 52.0696              |         124.1726      |
| FasfPitch | 64         | -                  | -                    |         84.8399       |   
