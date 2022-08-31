# WaveGlow模型-推理指导


- 概述

- 推理环境准备

- 快速上手

  - 获取源码
  - 准备数据集
  - 模型推理

- 模型推理性能

- 配套环境

## 概述

WaveGlow是一款用于语音合成的基于流的生成网络，是一种基于流的网络，能够从梅尔谱图生成高质量的语音。WaveGlow 融合了Glow和WaveNet的原理，以提供快速、高效和高质量的音频合成，无需自动回归。WaveGlow 仅使用单个网络实现，仅使用单个成本函数进行训练同时也最大化了训练数据的可能性，这使得训练过程简单稳定。


- 参考实现：

  ```
  url=https://github.com/NVIDIA/waveglow
  commit_id=8afb643df59265016af6bd255c7516309d675168
  model_name=waveglow
  ```

  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

### 输入输出数据

- 输入数据

  | 输入数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- | -------- | --------------------------- | ------------ |
  | input    | wav      | batch_size x 80 x audio_seq | ND           |


- 输出数据

  | 输出数据 | 大小          | 数据类型 | 数据排布格式 |
  | -------- | ------------- | -------- | ------------ |
  | output1  | 1 x audio_seq | BIN      | ND           |

## 推理环境准备

该模型需要以下插件与驱动

- **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.8.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

## 快速上手

### 1.安装依赖

```
git clone https://github.com/NVIDIA/waveglow.git
cd waveglow
git submodule init
git submodule update
git apply ../WaveGlow.patch
cd ..
# 安装依赖
pip3 install -r requirements.txt
```

### 2.准备数据集

1. 下载[LJSpeech-1.1数据集](https://gitee.com/link?target=https%3A%2F%2Fdata.keithito.com%2Fdata%2Fspeech%2FLJSpeech-1.1.tar.bz2)，解压至data目录

   ```0
   wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
   mkdir data
   tar jxvf LJSpeech-1.1.tar.bz2 ./data/
   ```

   解压后数据集目录结构如下:

   ```
       data
       |-- LJSpeech-1.1
           |-- wavs
           |    |-- LJ001-0001.wav
           |    |-- LJ001-0002.wav
           |    |-- …
           |    |-- LJ050-0278
           |-- metadata.csv
       |-- README
   ```

2. 数据预处理。

   ```
   # 测试集为LJSpeech-1.1数据集中前10条数据
   ls ./data/LJSpeech-1.1/wavs/*.wav | head -n10 > test_files.txt
   # 运行我方提供的数据预处理python文件
   python3 WaveGlow_preprocess.py -f test_files.txt -c waveglow/config.json -o ./prep_data/
   ```

获得数据处理结果``./prep_data/*.bin``和``./prep_data/*.txt``

### 3.模型推理

1. 模型转换

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      ```
      # 获取pt文件，保存到当前目录中
      wget https://drive.google.com/file/d/1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF/view 
      ```

   2. 导出onnx文件。

      1. 使用WaveGlow_pth2onnx.py导出onnx文件

         ```
         python3 WaveGlow_pth2onnx.py -i ./waveglow_256channels_universal_v5.pt -o ./
         ```

         获得WaveGlow_onnx.onnx文件

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称型号（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名(${chip_name}=Ascend310P3)
         回显如下：
         +--------------------------------------------------------------------------------------------+
         | npu-smi 22.0.0                       Version: 22.0.2                                       |
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 16.5         55                0    / 0              |
         | 0       0         | 0000:5E:00.0    | 0            931  / 21534                            |
         +===================+=================+======================================================+
         ```

      3. 执行ATC命令

         ```
         # 切换到onnx的保存目录
         cd ../infer/
         # 执行atc命令
         atc --model=WaveGlow.onnx \
             --output=WaveGlow \
             --input_shape="mel:1,80,-1" \
             --framework=5 \
             --input_format=ND \
             --soc_version=Ascend${chip_name} \
             --log=debug \
             --dynamic_dims="154;164;443;490;651;699;723;760;832;833"
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：上述命令查找得到的处理器型号。
           -   --dynamic_dims：ND格式下动态维度的档位。

         运行成功后生成WaveGlow.om模型文件。

2.开始推理验证

a. 使用ais-infer工具进行推理。


ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]


b. 执行推理

运行 WaveGlow_ais_infer 脚本。

```
mkdir out
python3 WaveGlow_ais_infer.py --ais_infer_path ${ais_infer_path} --bs 1
```

- 参数说明：

  -   --ais_infer_path：ais-infer推理脚本`ais_infer.py`所在路径，如“./tools/ais-bench_workload/tool/ais_infer/”。

  推理后的输出默认在当前目录out下。

  >**说明：** 
  >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见[《ais_infer 推理工具使用文档》](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)

c. 精度验证

通过主观听生成'.wav'音频文件验证模型的精度。

执行WaveGlow_postprocess.py脚本对ais_infer推理结果进行后处理，得到'.wav'音频文件。

```
#10个文件同时转成音频，保存在./wav目录中
python WaveGlow_postprocess.py -f ./out -o ./wav
```

d.  性能验证

可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

```
python3 ${ais_infer_path}/ais_infer.py --model=./WaveGlow.om --dymDims=mel:1,80,699 --loop=300 --batchsize=1
```


## 模型推理性能

调用ACL接口推理计算，性能参考下列数据。

不同设备的吞吐率性能（fps）

| 芯片型号 | Batch Size | 数据集 | 精度 | 性能 |
| -------- | ---------- | ------ | ---- | ---- |
| batch1   | 0.65       | 0.23   |      | 2.82 |

310P3性能达到T4性能的2.82倍。