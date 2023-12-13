# Tacotron2_dyn模型-推理指导

- [tacotron2_dyn模型-推理指导](#wav2vec2模型-推理指导)
- [概述](#概述)
  - [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
  - [安装CANN包](#安装cann包)
  - [安装Ascend-cann-aie](#安装ascend-cann-aie)
  - [安装Ascend-cann-torch-aie](#安装ascend-cann-torch-aie)
- [快速上手](#快速上手)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
- [模型推理性能\&精度](#模型推理性能精度)

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


## 输入输出数据
- 输入数据

  | 输入数据 | 数据类型 |        大小         | 数据排布格式 |
  | :------: | :------: | :-----------------: | :----------: |
  |   seq    |  INT64   | batchsize x seq_len |      ND      |
  | seq_lens |  INT32   |      batchsize      |      ND      |

- 输出数据

  | 输出数据 | 数据类型 |        大小         | 数据排布格式 |
  | :------: | :------: | :-----------------: | :----------: |
  |   wavs   | FLOAT32  | batchsize x wav_len |      ND      | |


# 推理环境准备

- 该模型需要以下依赖

  **表 1**  版本配套表


| 配套                      | 版本       |
| ------------------------- | --------------- |
| CANN                      | 7.0.RC1.alpha003   |
| Python                    | 3.9        | -                          |              |
| PyTorch （cuda）                   | 2.0.1      |
| torchAudio   (cuda)           | 2.0.1       |
| Ascend-cann-torch-aie | 6.3.T200           
| Ascend-cann-aie       | 6.3.T200        
| 芯片类型                  | Ascend310P3     | 


## 安装CANN包

 ```
 chmod +x Ascend-cann-toolkit_7.0.RC1.alpha003_linux-x86_64.run
./Ascend-cann-toolkit_7.0.RC1.alpha003_linux-x86_64.run --install
 ```

## 安装Ascend-cann-aie
1. 安装
```
chmod +x ./Ascend-cann-aie_${version}_linux-${arch}.run
./Ascend-cann-aie_${version}_linux-${arch}.run --check
# 默认路径安装
./Ascend-cann-aie_${version}_linux-${arch}.run --install
# 指定路径安装
./Ascend-cann-aie_${version}_linux-${arch}.run --install-path=${AieInstallPath}
```
2. 设置环境变量
```
cd ${AieInstallPath}
source set_env.sh
```
## 安装Ascend-cann-torch-aie
1. 安装
 ```
# 安装依赖
conda create -n py39_pt2.0 python=3.9.0 -c pytorch -y
conda install decorator -y
pip install attrs
pip install scipy
pip install synr==0.5.0
pip install tornado
pip install psutil
pip install cloudpickle
wget https://download.pytorch.org/whl/cpu/torch-2.0.1%2Bcpu-cp39-cp39-linux_x86_64.whl
pip install torch-2.0.1+cpu-cp39-cp39-linux_x86_64.whl

# 解压
tar -xvf Ascend-cann-torch-aie-${version}-linux-${arch}.tar.gz
cd Ascend-cann-torch-aie-${version}-linux-${arch}

# C++运行模式
chmod +x Ascend-cann-torch-aie_${version}_linux-${arch}.run
# 默认路径安装
./Ascend-cann-torch-aie_${version}_linux-${arch}.run --install
# 指定路径安装
./Ascend-cann-torch-aie_${version}_linux-${arch}.run --install-path=${TorchAIEInstallPath}

# python运行模式
pip install torch_aie-${version}-cp{pyVersion}-linux_x86_64.whl
 ```
 > 说明：如果用户环境是[libtorch1.11](https://download.pytorch.org/libtorch/cu113/libtorch-shared-with-deps-1.11.0%2Bcu113.zip)，需要使用配套的torch 1.11-cpu生成torchscript，再配套使用torch-aie-torch1.11的包。

2. 设置环境变量
```
cd ${TorchAIEInstallPath}
source set_env.sh
```



# 快速上手

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

   ```bash
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
   ```

   ```
   pip3 install -r requirements.txt
   ```



## 准备数据集
- 该模型使用`LJSpeech`数据集进行精度评估，`Pytorch`源码仓下已包含验证数据（500条文本数据），文件结构如下：
   ```
   filelists
   └── ljs_audio_text_test_filelist.txt
   ```


## 模型推理

1. 获取权重文件  
   `nvidia_tacotron2pyt_fp32_20190427`：[下载地址](https://ngc.nvidia.com/catalog/models/nvidia:tacotron2_pyt_ckpt_fp32)  
   `nvidia_waveglowpyt_fp32_20190427`：[下载地址](https://ngc.nvidia.com/catalog/models/nvidia:waveglow_ckpt_fp32)  
   将下载的模型文件`nvidia_tacotron2pyt_fp32_20190427`和`nvidia_waveglowpyt_fp32_20190427`放在新建的`checkpoints`文件夹下。

2. 生成torchscript模型 

    1. 先将tensorrt/tacotron2_torch2ts.py 和 tensorrt/waveglow_torch2ts.py脚本放在模型源码 /DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2 路径下

    2. 修改源码：

      - 删掉/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/tacotron2/model.py 里面第230行， （删掉Encoder类里面infer函数前面的@torch.jit.export）

    3. 执行以下命令生成两个torchscript模型： tacotron2， waveglow
    ```
    cd ./DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2

    python3 tensorrt/tacotron2_torch2ts.py --tacotron2 ./checkpoints/nvidia_tacotron2pyt_fp32_20190427 -o ./tacotron2dyn/ -bs 1

    python3 tensorrt/waveglow_torch2ts.py --waveglow ./checkpoints/nvidia_waveglowpyt_fp32_20190427 -o ./tacotron2dyn/ --config-file config.json

     ```
    - 参数说明：
         - --tacotron2, -– waveglow：模型权重
         - -o：torchscript模型输出文件夹路径, 请按需修改
         - -bs: batch size


3. 模型编译

   ```bash
   # 执行compile.py后经过torch-aie编译后的ts模型会保存到指定输出路径

   python compile.py --encoder_path ./tacotron2dyn/traced_tacotron2dyn_encoder.ts --decoder_path ./tacotron2dyn/traced_tacotron2dyn_decoder.ts --posnet_path ./tacotron2dyn/traced_tacotron2dyn_posnet.ts --waveglow_path ./tacotron2dyn/traced_waveglow.ts --compiled_models_folder ./tacotron2dyn/compiled_models --batchsize 1
   ```
      - 参数说明：
         - --encoder_path, --decoder_path, --posnet_path：由于tacotron模型需要分成三个部分分别编译， 此为各自torchscript模型路径。--waveglow_path 为waveglow模型的torchscript输入路径
         - --compiled_models_folder ：模型编译后数据输出文件夹路径。执行该命令后将在此路径下生成四个对应的ts模型文件
         - --batchsize ：batch_size。根据实际情况修改


4. 模型推理性能验证

    由于模型推理为多个子模型串联，仅测量单个子模型性能没有意义， 因此性能通过计算多个子模型推理的总耗时测得
   ```bash
   # 执行perf_test.py后对应batch_size的模型性能会打印到终端
    python perf_test.py --encoder_model_path ./tacotron2dyn/compiled_models/bs1/encoder_model.ts --decoder_model_path ./tacotron2dyn/compiled_models/bs1/decoder_model.ts --posnet_model_path ./tacotron2dyn/compiled_models/bs1/posnet_model.ts --batch_size 1
   ```
   - 参数说明：
      - --encoder_model_path, --decoder_model_path,  --posnet_model_path: 各个子模型经过torch-aie编译后的ts模型路径
      - --batch_size ：batch_size。根据实际情况修改

5. 模型推理精度验证

    先将inference_acc.py 脚本放在模型源码 /DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2 路径下

   ```
    cd ./tacotron2dyn/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2

    python inference.py --encoder_model_path ./tacotron2dyn/compiled_models/bs1/encoder_model.ts --decoder_model_path ./tacotron2dyn/compiled_models/bs1/decoder_model.ts --posnet_model_path ./tacotron2dyn/compiled_models/bs1/posnet_model.ts  --waveglow_model_path ./tacotron2dyn/compiled_models/bs1/waveglow_model.ts --input ./tacotron2dyn/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/filelists/ljs_audio_text_test_filelist.txt -bs 1 --gen_wave --max_input_len 50

   ```
   生成的wav文件默认放在./DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/output/audio 目录下
   - 参数说明：
      - --encoder_model_path, --decoder_model_path, --posnet_model_path, --waveglow_model_path: 各个子模型经过torch-aie编译后的ts模型路径
      - --input： 数据集路径
      - -bs ：batch_size。根据实际情况修改
      - --gen_wave：添加此参数时将推理waveglow生成wav文件， 用于人工判断语音质量
      - --max_input_len： 输入序列最大长度为50


# 模型推理性能&精度

   | NPU芯片型号 | Batch Size |  数据集   |  精度 |  性能| 
   | :-------:  | :--------: | :------: | :-----:           | :----: |
   |Ascend310P3 |      1     | LJSpeech |  人工判断语音质量  |416   |  
   |Ascend310P3 |      4     | LJSpeech |  人工判断语音质量  |1455  |  

