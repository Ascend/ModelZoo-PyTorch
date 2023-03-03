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
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.6.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/NVIDIA/DeepLearningExamples
   cd ./DeepLearningExamples
   git checkout master
   git reset --hard 6610c05c330b887744993fca30532cbb9561cbde
   mv ../p1.patch ./
   patch -p1 < p1.patch
   cd ..
   git clone https://github.com/NVIDIA/dllogger.git
   cd ./dllogger
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
   
   以上步骤均执行下面指令完成：
   
   ```
   python3 DeepLearningExamples/PyTorch/SpeechSynthesis/FastPitch/prepare_dataset.py --wav-text-filelists DeepLearningExamples/PyTorch/SpeechSynthesis/FastPitch/filelists/ljs_audio_text_val.txt --n-workers 16 --batch-size 1 --dataset-path ./LJSpeech-1.1 --extract-mels --f0-method pyin
   ```
   - 参数说明：
      -   --wav-text-filelists：包含数据集文件路径的txt文件
      -   --n-workers：使用的CPU核心数
      -   --batch-size：批次数
      -   --dataset-path：数据集路径
      -   --extract-mels：默认参数
      -   --f0-method：默认参数，代码中只包含了pyin选项，不可替换



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      ```
      wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/FastPitch/PTH/nvidia_fastpitch_210824.pt
      wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/FastPitch/PTH/nvidia_waveglow256pyt_fp16.pt
      ```
      （waveglow为语音生成器，不在本模型范围内, 但为了确保代码能正常运行，需要下载）

   2. 获取pt输出
      为了后面推理结束后将om模型推理精度与原pt模型精度进行对比，脚本运行结束会在test文件夹下创建mel_tgt_pth用于存放pth模型输入数据，mel_out_pth用于存放pth输出数据，input_bin用于存放二进制数据集，input_bin_info.info用于存放二进制数据集的相对路径信息

      ```
      python3 data_process.py -i phrases/tui_val100.tsv --dataset-path=./LJSpeech-1.1 --fastpitch ./nvidia_fastpitch_210824.pt --waveglow ./nvidia_waveglow256pyt_fp16.pt
      ```
      - 参数说明：
         -   -i：保存数据集文件的路径的tsv文件
         -   -o：输出二进制数据集路径
         -   --dataset-path：数据集路径
         -   --fastpitch：fastpitch权重文件路径
         -   --waveglow：waveglow权重文件路径

   3. 导出onnx文件。

      1. 使用pth2onnx.py导出onnx文件。

         ```
         python3 pth2onnx.py -i phrases/tui_val100.tsv --fastpitch nvidia_fastpitch_210824.pt --waveglow nvidia_waveglow256pyt_fp16.pt --energy-conditioning --batch-size 1
         ```

         获得FastPitch.onnx文件。

      2. 优化ONNX文件。

         ```
         python3 -m onnxsim FastPitch.onnx FastPitch_sim.onnx --dynamic-input-shape --input-shape 1,200
         ```

         获得FastPitch_sim.onnx文件。

         - 参数说明：
            -   FastPitch.onnx：原onnx模型文件
            -   FastPitch_sim.onnx：onnxsim生成的简化onnx模型文件
            -   --dynamic-input-shape：动态shape
            -   --input-shape：输入的shape(batchsize,200)


   4. 使用ATC工具将ONNX模型转OM模型。

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

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 配置环境变量。

      ```
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      ```

   3. 执行推理。

        ```
        python3 -m ais_bench --model FastPitch_bs1.om --input test/input_bin --output result --outfmt BIN
        ```

        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入数据路径。
             -   --output：推理结果输出路径。
             -   --outfmt：推理结果输出格式。

        推理后的输出默认在当前目录result下。


   4. 精度验证。

      调用脚本分别对比input中创建的mel_tgt_pth输入数据和ais_bench推理结果./result/{}，以及pthm模型mel_out_pth输出数据，可以分别获得om和pth模型的Accuracy数据。

      ```
      python3 infer_test.py ./result/{}
      ```
      -   参数说明：
         -   ./result/{}：ais_bench推理结果保存路径


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



| 芯片型号 | Batch Size | 数据集| 性能|  om精度    |   pth精度      
| --------- | ----| ----------|---------|   ------   |--------
| 310P3 |  1       | LJSpeech-1.1 |   202.265      |    mel_loss:11.260     |     mel_loss:13.400
| 310P3 |  4       | LJSpeech-1.1t |    249.906      |  mel_loss:11.260       |  mel_loss:13.400
| 310P3 |  8       | LJSpeech-1.1 |  240.211     |      mel_loss:11.260      |mel_loss:13.400
| 310P3 |  16       | LJSpeech-1.1 |   239.614      |   mel_loss:11.260      | mel_loss:13.400
| 310P3 |  32       | LJSpeech-1.1 |    232.589      |  mel_loss:11.260     |mel_loss:13.400
| 310P3 |  64       | LJSpeech-1.1 |  214.174     |     mel_loss:11.260      |mel_loss:13.400