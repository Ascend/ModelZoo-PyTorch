# HiFiGAN-推理指导

- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
- [模型推理性能&精度](#模型推理性能&精度)

******


# 概述
HiFiGAN是一种基于GAN的声码器，HiFiGAN同时拥有多尺度判别器（Multi-Scale Discriminator，MSD）和多周期判别器（Multi-Period Discriminator，MPD），可以尽可能增强GAN判别器鉴别音频真伪的能力。  
同时，HiFi-AN生成器中采用了多感受野融合模块。相比WaveNet为了增大感受野，采用叠加空洞卷积，逐样本点生成的方法，虽然音质提升，但是模型较大，推理速度较慢。HiFiGAN则提出了一种残差结构，交替使用空洞卷积和普通卷积增大感受野，在保证合成音质的同时，提高了推理速度。

- 版本说明：
  ```
  url=https://github.com/jik876/hifi-gan
  commit_id=4769534d45265d52a904b850da5a622601885777
  model_name=HifiGAN
  ```

### 输入输出数据

- 输入数据

  | 输入数据 | 数据类型  |            大小            | 数据排布格式 | 
  |:-----:|:------------------------:|:------:|:----------:| 
  | mel_spec    | FLOAT32 | batchsize x 80 x mel_len |  ND    |


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
   git clone https://github.com/jik876/hifi-gan.git
   cd hifi-gan
   git reset --hard 4769534d45265d52a904b850da5a622601885777
   mkdir -p output output/gen_wavs  # 新建output文件夹，作为模型结果的默认保存路径
   ```
   
2. 安装依赖  
   ```
   pip3 install -r requirements.txt
   ```

3. 获取`OM`推理代码  
   将推理部署代码放在`hifi-gan`源码仓目录下。
   ```
    HiFiGAN_for_PyTorch
    ├── pth2onnx.py        放到hifi-gan下
    ├── atc.sh             放到hifi-gan下
    └── om_val.py          放到hifi-gan下
   ```   


## 准备数据集
- 该模型使用`LJSpeech`数据集进行精度评估，下载[LJSpeech数据集](https://keithito.com/LJ-Speech-Dataset/)，将音频数据放到`LJSpeech-1.1/wavs`文件下，文件结构如下：
   ```
   LJSpeech-1.1
   ├── training.txt
   ├── validation.txt
   └── wavs
      ├── LJ001-0001.wav
      ├── LJ001-0002.wav
      ├── ……
      └── LJ050-0278.wav
   ```


## 模型推理
### 1 模型转换  
将模型权重文件`.pth`转换为`.onnx`文件，再使用`ATC`工具将`.onnx`文件转为离线推理模型`.om`文件。

1. 获取权重文件  
   下载[权重文件及相应的配置文件](https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing)，以`generator_v1`为例，
   将下载的模型文件`generator_v1`和配置文件`config.json`放在`hifi-gan`目录下。

2. 导出`ONNX`模型  
   运行`pth2onnx.py`导出`ONNX`模型，结果默认保存在`output`文件夹下。  
   ```
   python3 pth2onnx.py --output_dir output \
                       --checkpoint_file generator_v1 \
                       --config_file config.json
   ```

3. 使用`ATC`工具将`ONNX`模型转为`OM`模型  
   3.1 配置环境变量  
   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
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
   运行`atc.sh`导出`OM`模型。
   ```
   atc --framework=5 --input_format=ND --log=error --soc_version=Ascend${chip_name} \
    --model=${output_dir}/${model}.onnx --output=${output_dir}/${model}_bs${bs} \
    --input_shape="mel_spec:${bs},80,1,-1" \
    --dynamic_dims="250;500;750;1000;1250;1500;1750;2000"
   ```
      - 参数说明
      ：
        -   `--model`：ONNX模型文件
        -   `--framework`：5代表ONNX模型
        -   `--output`：输出的OM模型
        -   `--input_shape`：输入数据的shape
        -   `--log`：日志级别
        -   `--soc_version`：处理器型号
        -   `--dynamic_dims`：设置ND格式下动态维度的档位

    
### 2 开始推理验证

1. 安装`ais_bench`推理工具  
   请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

2. 执行推理  
   运行`om_val.py`推理OM模型，合成语音默认保存在`output/wavs`文件夹下。
   ```
   python3 om_val.py --input_wavs_dir LJSpeech-1.1/wavs \
                     --output_wavs_dir output/gen_wavs \
                     --om output/generator_v1_bs1.om \
                     --config_file config.json --batch 1
   ```

3. 性能验证  
   可使用`ais_bench`推理工具的纯推理模式验证不同`batch_size`的`OM`模型的性能，参考命令如下：
   ```
   python3 -m ais_bench --model output/generator_v1_bs${bs}.om --loop 1000 --batchsize ${bs} --dymDims "mel_spec:${bs},80,1,${mel_len}" --outputSize "10000000"
   ```
   其中，`bs`为模型`batch_size`，`mel_len`为输入数据的长度。

# 模型推理性能&精度

调用ACL接口推理计算，性能&精度参考下列数据。

|   芯片型号   | Batch Size | mel_len |   数据集     |     精度      |     性能     |
|:-----------:|:----------:|:-------:|:--------:|:------:|:----------:|
| Ascend310P3 |     1      |   250   |  LJSpeech   | 人工判断语音质量 | 339.62 fps |
| Ascend310P3 |     1      |   500   |  LJSpeech   | 人工判断语音质量 | 248.54 fps |
| Ascend310P3 |     1      |   750   |  LJSpeech   | 人工判断语音质量 | 159.83 fps |
| Ascend310P3 |     1      |  1000   |  LJSpeech   | 人工判断语音质量 | 121.50 fps |
| Ascend310P3 |     8      |   250   |  LJSpeech   | 人工判断语音质量 | 637.95 fps |
| Ascend310P3 |     8      |   500   |  LJSpeech   | 人工判断语音质量 | 300.21 fps |
| Ascend310P3 |     8      |   750   |  LJSpeech   | 人工判断语音质量 | 191.81 fps |
| Ascend310P3 |     8      |  1000   |  LJSpeech   | 人工判断语音质量 | 139.74 fps |
- 说明：由于音频数据输入长度不同，故给出不同mel_len的性能数据作为参考。