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

| 配套         | 版本               | 环境准备指导                                                 |
| ----------- |------------------| ------------------------------------------------------------ |
| 固件与驱动      | 22.0.3           | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN        | 7.0.RC1.alpha003 | -                                                            |
| Python      | 3.9.11           | -                                                            |
| PyTorch     | 2.0.1            | -                                                            |
| Torch_AIE  | 6.3.rc2          | \                                                            |
| 芯片类型       | Ascend310P3      | \                                                            |


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

3. 获取`AIE`推理代码  
   将推理部署代码放到`Pytorch`源码相应目录下。
   ```
    FastSpeech2_for_PyTorch
    ├── pth2aie.py        放到FastSpeech2下
    └── aie_val.py          放到FastSpeech2下
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

1. 获取权重文件  
   下载FastSpeech2[权重文件](https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F?usp=sharing)，
   将下载的模型文件`900000.pth.tar`放在新建的`output/ckpt/LJSpeech`文件夹下。
2. 下载gen_core[文件](https://gitee.com/liurf_hw/om_gener/tree/master/gen_core)到当前目录

2. 导出`AIE`模型  
   运行`pth2aie.py`导出`aie`模型，结果默认保存在`output/`文件夹下。  
   ```
   python3 pth2aie.py --restore_step 900000 \
                    -p config/LJSpeech/preprocess.yaml \
                    -m config/LJSpeech/model.yaml \
                    -t config/LJSpeech/train.yaml
   ```

    
### 2 开始推理验证
1. 执行推理  
   运行`aie_val.py`推理OM模型，性能结果文件保存在`result.txt`中，合成语音默认保存在`output/result/LJSpeech`文件夹下。可设置参数`-vp/-ve/-vd`分别调整合成语音的`pitch`（音调）/`energy`（响度）/`duration`（语速）/`aie_path` （om保存路径）。
   ```
   python3 aie_val.py --source preprocessed_data/LJSpeech/val.txt \
               -p config/LJSpeech/preprocess.yaml \
               -t config/LJSpeech/train.yaml \
               -vp 1.0 -ve 1.0 -vd 1.0 \
               --batch 1 --device_id 0
   ```

# 模型推理性能&精度

调用ACL接口推理计算，性能&精度参考下列数据。

|   芯片型号   | Batch Size |    数据集     |     精度      |      性能      |
|:-----------:|:-------------:|:-----------:|:--------:|:------------:|
| Ascend310P3 |     1      |  LJSpeech   | 人工判断语音质量 | 13.28 wavs/s |
- 说明：由于模型推理为多个子模型串联，故性能采用端到端推理LJSpeech验证集中512条文本数据测得。
