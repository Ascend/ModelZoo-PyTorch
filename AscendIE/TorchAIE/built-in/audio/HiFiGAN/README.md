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

| 配套           | 版本               | 环境准备指导                                                 |
|--------------|------------------| ------------------------------------------------------------ |
| 固件与驱动        | 23.0.RC1         | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN         | 7.0.RC1.alpha003 | -                                                            |
| Python       | 3.9.11           | -                                                            |
| PyTorch      | 2.0.1            | -                                                            |
| Torch_AIE    | 6.3.rc2          | \                                                            |
| 芯片类型         | Ascend310P3      | \                                                            |


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

3. 获取`AIE`推理代码  
   将推理部署代码放在`hifi-gan`源码仓目录下。
   ```
    HiFiGAN_for_PyTorch
    ├── pth2aie.py            放到hifi-gan下
    ├── aie_val.py            放到hifi-gan下
    └── aie_perf.py           放到hifi-gan下
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
执行模型编译脚本，导出aie模型。

1. 获取权重文件  
   下载[权重文件及相应的配置文件](https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing)，以`generator_v1`为例，
   将下载的模型文件`generator_v2`和配置文件`config.json`放在`hifi-gan`目录下。

2. 导出`aie`模型  
   运行`pth2aie.py`导出`AIE`模型，结果默认保存在`output`文件夹下。  
   ```
   python pth2aie.py  --batch_size 1 --device_id 0 --mel_len 1000 --output_dir output --checkpoint_file generator_v2 --config_file config.json --aie_dir aie_batch1.ts --dynamic_dim True
   ```

    
### 2 开始推理验证

2.1. 执行推理  
   由于不同torch版本兼容问题，需要手动修改数据的处理方式：打开meldataset.py，原代码64-65修改为以下内容
   ```
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
        center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
   ```
2.2. 运行`aie_val.py`推理，合成语音默认保存在`output/wavs`文件夹下。
   ```
   python3 aie_val.py --input_wavs_dir LJSpeech-1.1/wavs \
                      --output_wavs_dir output/gen_wavs \
                      --aie_dir ./aie_all_dim.pt          \
                      --config_file config.json --batch 1
   ```

2.3. 性能验证  
   运行`aie_perf.py`，输出性能结果，参考命令如下：
   ```
   python aie_perf.py --batch_size=8 --mel_len=1000 --aie_dir=aie_batch1.ts
   ```
   其中，`batch_size`为模型`batch_size`，`mel_len`为输入数据的长度。

# 模型推理性能&精度

调用ACL接口推理计算，性能&精度参考下列数据。

|   芯片型号   | Batch Size | mel_len |   数据集     |     精度      |     性能     |
|:-----------:|:----------:|:-------:|:--------:|:------:|:----------:|
| Ascend310P3 |     1      |   250   |  LJSpeech   | 人工判断语音质量 | 315.85 fps |
| Ascend310P3 |     1      |   500   |  LJSpeech   | 人工判断语音质量 | 226.84 fps |
| Ascend310P3 |     1      |   750   |  LJSpeech   | 人工判断语音质量 | 154.65 fps |
| Ascend310P3 |     1      |  1000   |  LJSpeech   | 人工判断语音质量 | 117.55 fps |
| Ascend310P3 |     8      |   250   |  LJSpeech   | 人工判断语音质量 | 517.81 fps |
| Ascend310P3 |     8      |   500   |  LJSpeech   | 人工判断语音质量 | 260.73 fps |
| Ascend310P3 |     8      |   750   |  LJSpeech   | 人工判断语音质量 | 175.77 fps |
| Ascend310P3 |     8      |  1000   |  LJSpeech   | 人工判断语音质量 | 132.70 fps |
- 说明：由于音频数据输入长度不同，故给出不同mel_len的性能数据作为参考。