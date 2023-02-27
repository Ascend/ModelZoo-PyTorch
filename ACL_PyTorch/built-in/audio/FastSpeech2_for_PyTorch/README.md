# FastSpeech2-推理指导

- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
- [模型推理性能&精度](#模型推理性能&精度)

******


# 概述
FastSpeech2是一种非自回归的语音合成网络。所谓自回归是指模型的输出依赖过去的输入，最典型的比如Tacotron2模型。自回归的特性会导致语音合成速度慢、稳健性不足、可控性缺乏等问题。  
为解决这些问题，FastSpeech2采用Transformer结构，输入音素序列，可以并行生成Mel频谱，大大加速合成速度。同时提出时间预测器来确保音素和Mel频谱之间的硬对齐，相比自回归的软对齐能有效避免错误传播。并且增加长度调节器来调节语速，提供了对语速和韵律的控制。

- 版本说明：
  ```
  url=https://github.com/ming024/FastSpeech2
  commit_id=d4e79eb52e8b01d24703b2dfc0385544092958f3
  model_name=FastSpeech2
  ```

### 输入输出数据

- 输入数据

  | 输入数据   | 数据类型  |      大小          | 数据排布格式 | 
  |:-----:|:-------------------:|:------:|:----------:| 
  | texts     | INT64 | batchsize x src_len |  ND    |
  | src_masks | BOOL  | batchsize x src_len |  ND    |
  | mel_masks | BOOL  | batchsize x mel_len |  ND    |

- 输出数据

  | 输出数据 |  数据类型   |         大小          | 数据排布格式 |
  |:-----------:|:-------------------:|:----------------:|:----------:|
  | wavs        | FLOAT32   | batchsize x wav_len |   ND       |


# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

| 配套                                                     | 版本      | 环境准备指导                                                 |
| ------------------------------------------------------- |---------| ------------------------------------------------------------ |
| 固件与驱动                                                | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                    | 6.0.RC1 | -                                                            |
| Python                                                  | 3.7.5   | -                                                            |
| PyTorch                                                 | 1.10.1  | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |


# 快速上手

## 获取源码

1. 获取`Pytorch`源码  
   ```
   git clone https://github.com/ming024/FastSpeech2.git
   cd FastSpeech2
   git reset --hard d4e79eb52e8b01d24703b2dfc0385544092958f3
   mkdir -p output/ckpt/LJSpeech output/log/LJSpeech output/result/LJSpeech   # 新建output文件夹，作为模型结果的默认保存路径
   unzip hifigan/generator_LJSpeech.pth.tar.zip -d hifigan/
   ```
   
2. 安装依赖  
   ```
   pip3 install -r requirements.txt
   ```

3. 获取`OM`推理代码  
   将推理部署代码放到`Pytorch`源码相应目录下。
   ```
    FastSpeech2_for_PyTorch
    ├── pth2onnx.py        放到FastSpeech2下
    ├── atc.sh             放到FastSpeech2下
    └── om_val.py          放到FastSpeech2下
   ```   


## 准备数据集
- 该模型使用`LJSpeech`数据集进行精度评估，`Pytorch`源码仓下已包含验证数据（512条文本数据），文件结构如下：
   ```
   preproessed_data
   ├── AISHELL3
   ├── LibriTTS
   └── LJSpeech
      ├── speakers.json
      ├── stats.json
      ├── train.txt
      └── val.txt
   ```


## 模型推理
### 1 模型转换  
将模型权重文件`.pth`转换为`.onnx`文件，再使用`ATC`工具将`.onnx`文件转为离线推理模型`.om`文件。

1. 获取权重文件  
   下载FastSpeech2[权重文件](https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F?usp=sharing)，
   将下载的模型文件`900000.pth.tar`放在新建的`output/ckpt/LJSpeech`文件夹下。

2. 导出`ONNX`模型  
   运行`pth2onnx.py`导出`ONNX`模型，结果默认保存在`output/onnx`文件夹下。  
   ```
   python3 pth2onnx.py --restore_step 900000 \
                    -p config/LJSpeech/preprocess.yaml \
                    -m config/LJSpeech/model.yaml \
                    -t config/LJSpeech/train.yaml
   ```

3. 使用`ATC`工具将`ONNX`模型转为`OM`模型  
   3.1 配置环境变量  
   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   export ENABLE_RUNTIME_V2=0
   ```
   > **说明：**  
     该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

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
        -   `--dynamic_dims`：设置ND格式下动态维度的档位

    
### 2 开始推理验证

1. 安装`ais_bench`推理工具  
   请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

2. 执行推理  
   运行`om_val.py`推理OM模型，合成语音默认保存在`output/result/LJSpeech`文件夹下。可设置参数`-vp/-ve/-vd`分别调整合成语音的`pitch`（音调）/`energy`（响度）/`duration`（语速）。
   ```
   python3 om_val.py --source preprocessed_data/LJSpeech/val.txt \
               -p config/LJSpeech/preprocess.yaml \
               -t config/LJSpeech/train.yaml \
               -vp 1.0 -ve 1.0 -vd 1.0 \
               --batch 1 --device_id 0
   ```

3. 性能验证  
   可使用`ais_bench`推理工具的纯推理模式验证不同`batch_size`的`OM`模型的性能，FastSpeech2包括多个子模型，各子模型测试性能的参考命令如下：
   ```
   python3 -m ais_bench --model output/om/encoder_bs${bs}.om --loop 20 --batchsize ${bs} --dymShape "texts:${bs},${seq_len};src_masks:${bs},${seq_len}" --outputSize "1000000"
   python3 -m ais_bench --model output/om/variance_adaptor_bs${bs}.om --loop 20 --batchsize ${bs} --dymShape "enc_output:${bs},${seq_len},256;src_masks:${bs},${seq_len};p_control:1;e_control:1;d_control:1" --outputSize "1000000,1000000"
   python3 -m ais_bench --model output/om/decoder_bs${bs}.om --loop 20 --batchsize ${bs} --dymShape "output:${bs},250,256;mel_masks:${bs},250" --outputSize "1000000"
   python3 -m ais_bench --model output/om/postnet_bs${bs}.om --loop 20 --batchsize ${bs} --dymShape "dec_output:${bs},250,256" --outputSize "1000000"
   python3 -m ais_bench --model output/om/hifigan_bs${bs}.om --loop 20 --batchsize ${bs} --dymDims "mel_output:${bs},250,80" --outputSize "1000000"
   ```
   其中，`bs`为模型`batch_size`，`seq_len`为输入音频的长度。

# 模型推理性能&精度

调用ACL接口推理计算，性能&精度参考下列数据。

|   芯片型号   | Batch Size |    数据集     |     精度      |     性能      |
|:-----------:|:-------------:|:-----------:|:--------:|:-----------:|
| Ascend310P3 |     1      |  LJSpeech   | 人工判断语音质量 | 11.9 wavs/s |
- 说明：由于模型推理为多个子模型串联，仅测量单个子模型性能没有意义，故性能采用端到端推理LJSpeech验证集中512条文本数据测得。
