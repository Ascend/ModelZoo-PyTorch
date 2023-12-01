# wav2vec2模型-推理指导

- [wav2vec2模型-推理指导](#wav2vec2模型-推理指导)
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

wav2vec2 是一个用于语音表示学习的自监督学习框架，它完成了原始波形语音的潜在表示并提出了在量化语音表示中的对比任务。对于语音处理，该模型能够在未标记的数据上进行预训练而取得较好的效果。在语音识别任务中，该模型使用少量的标签数据也能达到最好的半监督学习效果。

- 参考实现：
  ```
  url = https://github.com/huggingface/transformers
  commit_id = 39b4aba54d349f35e2f0bd4addbe21847d037e9e
  model_name= wav2vec2
  ```


## 输入输出数据

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FP32 | batchsize x 100000 | ND         |


- 输出数据

  | 输出数据 | 数据类型     | 大小 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output  | FP32 | batchsize x 312 x 32 | ND        |


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

1. 获取源码

   通过Git获取对应版本的代码并安装的方法如下：
   ```bash
   git clone https://github.com/huggingface/transformers.git    # 克隆仓库的代码
   cd transformers                                              # 切换到模型的代码仓目录
   git checkout v4.20.0                                         # 切换到对应版本
   git reset --hard 39b4aba54d349f35e2f0bd4addbe21847d037e9e    # 将暂存区与工作区都回到上一次版本
   pip3 install ./                                              # 通过源码进行安装
   cd ..
   ```

2. 安装依赖

   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
   ```

   ```
   pip3 install -r requirements.txt
   ```
   > 如果报错： `OSError: sndfile library not found`， 则需要执行此命令： `sudo apt-get install libsndfile1`



## 准备数据集

1. 获取原始数据集

   本项目使用的数据是 [LibriSpeech](http://www.openslr.org/12) 数据集, 许可证信息：[CC BY 4.0](https://openslr.org/12/)。 其中推理使用的是 [test-clean](https://openslr.magicdatatech.com/resources/12/test-clean.tar.gz) 部分。

   ```bash
   wget https://openslr.magicdatatech.com/resources/12/test-clean.tar.gz

   # 解压到当前目录
   tar -xzvf test-clean.tar.gz
   ```

2. 数据预处理

   将原始数据集转换为模型输入的数据,执行```wav2vec2_preprocess.py```脚本，完成预处理。

   ```bash
   # 设置Batch size，请按需修改
   bs=8

   python3 wav2vec2_preprocess.py --input=${dataset_path} --batch_size=${bs} --output="./data/bin_om_input/bs${bs}/"
   ```
   - 参数说明:
     - --input: 测试数据集解压后的文件夹，这里即是 LibriSpeech 测试数据集解压后的路径。
     - --ouput: 测试数据集处理后保存bin文件的文件夹。
     - --batch_size: 数据集需要处理成的批次大小。

   预处理成功后生成的文件:

   - `data/bin_om_input/bs16`: 预处理的输出文件夹，是预处理后bin文件存放的目录，且一批数据处理成一个bin文件，作为模型推理的输入
   - `data/batch_i_filename_map_bs16.json`: 存放元数据，保存着每个批里面包含了哪些文件，即一个batch对应一个（batch为1）或多个（batch不为1）bin文件名
   - `data/ground_truth_texts.txt`: 存放每个样本的真实文本数据，即一段语音文件的文件名对应的真实文本数据

## 模型推理
1. 获取权重文件。
   
   1. 获取 Pytorch 模型配置文件和数据集词典文件等, 可通过 `wget` 或其它方式下载，以下以 `wget` 为例：
      ```bash
      mkdir wav2vec2_pytorch_model
      cd wav2vec2_pytorch_model

      wget https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/config.json
      wget https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/feature_extractor_config.json
      wget https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/preprocessor_config.json
      wget https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/special_tokens_map.json
      wget https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/tokenizer_config.json
      wget https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/vocab.json

      ```

   2. 获取模型权重文件。

      [模型权重文件](https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/pytorch_model.bin)使用官方开源的pyTorch模型， 必须将该权重文件下载到`wav2vec2_pytorch_model` 目录下和模型配置文件、词典文件等放在一起，然后回到主目录执行其它脚本。
      ```bash
      wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/pytorch_model.bin
      cd ..
      ```


2. 生成torchscript模型

   ```bash
   # 生成wav2vec2 torchscript 模型traced.pth
   python3 wav2vec2_pth2ts.py --pytorch_model_dir="./wav2vec2_pytorch_model" 
   ```


3. 执行模型编译和推理脚本（包含推理性能测试）

   ```bash
   # 执行compile.py后对应batch_size的模型性能会打印到终端
   python compile.py --batch_size 8 --input_path "./data/bin_om_input/bs8" --output_path "./data/bin_om_output/bs8" --torchscript_path "./traced.pth"
   ```
      - 参数说明：
         - --input_path：推理前输入数据路径
         - --output_path ：推理后数据输出路径
         - --torchscript_path: trace 之后保存的torchscript模型路径traced.pth


4. 模型推理的精度验证
   ```bash
   # 执行wav2vec2_postprocess.py后对应batch_size的模型精度会打印到终端， 并与基准精度比较
   python3 wav2vec2_postprocess.py --input "./data/bin_om_output/bs8" --batch_size 8 --data_path "./data"
   ```
   - 参数说明：
      - --input：推理后的输出数据路径
      - --data_path : 数据集路径






# 模型静态推理性能&精度

   | NPU芯片型号 | Batch Size |  数据集   |  精度 |  性能(有ts模型优化)| 性能(无ts模型优化)|
   | :-------:  | :--------: | :---------: | :-----: | :----: | :----: |
   |Ascend310P3 |      1     | LibriSpeech |  0.970  |86.58|  83.06    |
   |Ascend310P3 |      4     | LibriSpeech |  0.970  |89.48|  89.25    |
   |Ascend310P3 |      8     | LibriSpeech |  0.970  |92.25|  86.03    |
   |Ascend310P3 |      16    | LibriSpeech |  0.970  |90.25|  84.99    |
   |Ascend310P3 |      32    | LibriSpeech |  0.970  |85.41|  80.47    |
   |Ascend310P3 |      64    | LibriSpeech |  0.970  |84.81|  80.34   |
