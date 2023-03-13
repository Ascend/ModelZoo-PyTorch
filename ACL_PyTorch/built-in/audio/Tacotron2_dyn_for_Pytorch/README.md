# Tacotron2_dyn推理指导

- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
- [模型推理性能&精度](#模型推理性能&精度)

******


# 概述
Tacotron2是由Google Brain在2017年提出来的一个End-to-End语音合成框架。模型从下到上可以看作由两部分组成：
1. 声谱预测网络：一个Encoder-Attention-Decoder网络，输入字符序列，用于预测梅尔频谱的帧序列。
2. 声码器（vocoder）：一个WaveNet的修订版，输入预测的梅尔频谱帧序列，用于生成时域波形。

- 版本说明：
  ```
  url=https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2
  commit_id=7ce175430ff9af25b040ffe2bceb5dfc9d2e39ad
  model_name=Tacotron2
  ```

### 输入输出数据

- 输入数据

  | 输入数据 | 数据类型 |        大小         | 数据排布格式 |
  | :------: | :------: | :-----------------: | :----------: |
  |   seq    |  INT64   | batchsize x seq_len |      ND      |
  | seq_lens |  INT32   |      batchsize      |      ND      |

- 输出数据

  | 输出数据 | 数据类型 |        大小         | 数据排布格式 |
  | :------: | :------: | :-----------------: | :----------: |
  |   wavs   | FLOAT32  | batchsize x wav_len |      ND      |


# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.10.1  | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |


# 快速上手

## 获取源码

1. 获取`Pytorch`源码  
   ```
   git clone https://github.com/NVIDIA/DeepLearningExamples.git
   cd DeepLearningExamples
   git reset --hard 7ce175430ff9af25b040ffe2bceb5dfc9d2e39ad
   cd PyTorch/SpeechSynthesis/Tacotron2
   mkdir -p output/audio  # 新建output文件夹，作为模型结果的默认保存路径
   mkdir checkpoints
   ```
   
2. 安装依赖  
   ```
   pip3 install -r requirements.txt
   ```

3. 获取`OM`推理代码  
   将推理部署代码放到`Pytorch`源码相应目录下。
   ```
   Tacotron2_dyn_for_PyTorch
    ├── cvt_tacotron2onnx.py  放到Tacotron2/tensorrt下
    ├── cvt_waveglow2onnx.py  放到Tacotron2/tensorrt下
    ├── atc.sh        放到Tacotron2下
    └── om_val.py     放到Tacotron2下
   ```


## 准备数据集
- 该模型使用`LJSpeech`数据集进行精度评估，`Pytorch`源码仓下已包含验证数据（500条文本数据），文件结构如下：
   ```
   filelists
   └── ljs_audio_text_test_filelist.txt
   ```


## 模型推理
### 1 模型转换  
将模型权重文件`.pth`转换为`.onnx`文件，再使用`ATC`工具将`.onnx`文件转为离线推理模型`.om`文件。

1. 获取权重文件  
   `nvidia_tacotron2pyt_fp32_20190427`：[下载地址](https://ngc.nvidia.com/catalog/models/nvidia:tacotron2_pyt_ckpt_fp32)  
   `nvidia_waveglowpyt_fp32_20190427`：[下载地址](https://ngc.nvidia.com/catalog/models/nvidia:waveglow_ckpt_fp32)  
   将下载的模型文件`nvidia_tacotron2pyt_fp32_20190427`和`nvidia_waveglowpyt_fp32_20190427`放在新建的`checkpoints`文件夹下。

2. 导出`ONNX`模型  
   运行`pth2onnx.py`导出`ONNX`模型，结果默认保存在`output/onnx`文件夹下。  
   ```
   python3 tensorrt/cvt_tacotron2onnx.py --tacotron2 ./checkpoints/nvidia_tacotron2pyt_fp32_20190427 -o output/onnx/ -bs 1
   python3 tensorrt/cvt_waveglow2onnx.py --waveglow ./checkpoints/nvidia_waveglowpyt_fp32_20190427 -o output/onnx/ --config-file config.json
   ```

3. 使用`ATC`工具将`ONNX`模型转为`OM`模型  
   3.1 配置环境变量  
   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```
   > **说明：**  
   > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

   3.2 执行命令查看芯片名称（得到`atc`命令参数中`soc_version`）
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

   3.3 执行ATC命令  
   运行`atc.sh`导出`OM`模型，默认保存在`output/om`文件夹下。
   ```
   bash atc.sh --soc Ascend310P3 --bs 1
   ```
      - `atc`命令参数说明（参数见`atc.sh`）：
        -   `--model`：ONNX模型文件
        -   `--framework`：5代表ONNX模型
        -   `--output`：输出的OM模型
        -   `--input_format`：输入数据的格式
        -   `--input_shape`：输入数据的shape
        -   `--log`：日志级别
        -   `--soc_version`：处理器型号
        -   `--input_shape_range`：指定模型输入数据的shape范围

    
### 2 开始推理验证

1. 安装`ais_bench`推理工具  
   请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

2. 执行推理  
   运行`om_val.py`推理OM模型，合成语音默认保存在`output/audio`文件夹下。
   ```
   # 推理tacotron2 om
   python3 val.py -i filelists/ljs_audio_text_test_filelist.txt -bs 1 -device_id 0
    
   # 推理waveglow生成wav文件
   python3 val.py -i filelists/ljs_audio_text_test_filelist.txt -o output/audio -bs 1 -device_id 0 --gen_wav
   ```
   其中，`bs`为模型`batch_size`，`device_id`设置推理用第几号卡。

3. 性能验证  
   可使用`ais_bench`推理工具的纯推理模式验证不同`batch_size`的`OM`模型的性能，Tacotron2包括多个子模型，各子模型测试性能的参考命令如下：
   ```
   python3 -m ais_bench --model output/om/encoder_dyn.om --loop 20 --batchsize ${bs} --dymShape "sequences:${bs},${seq_len};sequence_lengths:${bs}" --outputSize "3000000,3000000,3000000"
   python3 -m ais_bench --model output/om/decoder_iter_dyn.om --loop 20 --batchsize ${bs} --dymShape "decoder_input:${bs},80;attention_hidden:${bs},1024;attention_cell:${bs},1024;decoder_hidden:${bs},1024;decoder_cell:${bs},1024;attention_weights:${bs},${seq_len};attention_weights_cum:${bs},${seq_len};attention_context:${bs},512;memory:${bs},${seq_len},512;processed_memory:${bs},${seq_len},128;mask:${bs},${seq_len}" --outputSize "20000,20000,20000,20000,20000,20000,20000,20000,20000"
   python3 -m ais_bench --model output/om/postnet_dyn.om --loop 20 --batchsize ${bs} --dymShape "mel_outputs:${bs},80,250" --outputSize "640000"
   ```
   其中，`bs`为模型`batch_size`，`seq_len`为输入音频的长度。

# 模型推理性能&精度

调用ACL接口推理计算，性能&精度参考下列数据。

|  芯片型号   | Batch Size |  数据集  |       精度       |    性能     |
| :---------: | :--------: | :------: | :--------------: | :---------: |
| Ascend310P3 |     1      | LJSpeech | 人工判断语音质量 | 636 wavs/s  |
| Ascend310P3 |     4      | LJSpeech | 人工判断语音质量 | 2076 wavs/s |
- 说明：由于模型推理为多个子模型串联，仅测量单个子模型性能没有意义，故性能采用端到端推理LJSpeech验证集中500条文本数据测得。