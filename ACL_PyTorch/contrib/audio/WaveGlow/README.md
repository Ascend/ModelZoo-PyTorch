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
  git clone https://github.com/NVIDIA/waveglow.git
  cd waveglow
  git submodule init
  git submodule update
  git apply ../WaveGlow.patch
  ```

​		注意最后需应用我方提供的WaveGlow.patch文件

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
# 创建conda环境
conda create --name WaveGlow python=3.7.5
conda activate WaveGlow
# 安装依赖
pip install -r requirment.txt
```

### 2.准备数据集

1. 下载[LJSpeech-1.1数据集](https://gitee.com/link?target=https%3A%2F%2Fdata.keithito.com%2Fdata%2Fspeech%2FLJSpeech-1.1.tar.bz2)，解压至data目录

   ```0
   wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
   tar jxvf LJSpeech-1.1.tar.bz2 ./${data_path}/
   ```

   解压后数据集目录结构如下:

   ```
   ${data_path}
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
   ls ./${data_path}/LJSpeech-1.1/wavs/*.wav | head -n10 > test_list.txt
   # 运行我方提供的数据预处理python文件
   python WaveGlow_preprocess.py -f test_list.txt -c config.json -o ../infer/data/
   ```

获得数据处理结果``../infer/data/LJSpeech-1.1/wavs/*.bin``和``../infer/data/LJSpeech-1.1/wavs/*.txt``

### 3.模型推理

1. 模型转换

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       ```
       # 获取pt文件，保存到waveglow目录中
       wget https://drive.google.com/file/d/1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF/view 
       ```
   
   2. 导出onnx文件。
   
      1. 使用WaveGlow_pth2onnx.py导出onnx文件
   
         ```
         python WaveGlow_pth2onnx.py -i ./waveglow_256channels_universal_v5.pt -o ../infer/
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
         atc --model=WaveGlow_onnx.onnx \
             --output=WaveGlow_om \
             --input_shape="mel:1,80,-1" \
             --framework=5 \
             --input_format=ND \
             --soc_version=${chip_name} \
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
           -   --dynamic_dims：设为10条测试集数据的shape
           
         
         运行成功后生成WaveGlow_om.om模型文件。

2.开始推理验证

 1. 查看[《ais_infer 推理工具使用文档》](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)，完成ais_infer工具安装：

    ```
    git https://gitee.com/ascend/tools.git
    cd ./tools/ais-bench_workload/tool/ais_infer/backend
    pip3.7 wheel ./
    pip3 install ./aclruntime-0.0.1-cp37-cp37m-linux_x86_64.whl
    ```

 2. 执行推理

    ```
    # 设置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    # 推理前使用 'npu-smi info' 命令查看 device 是否在运行其它推理任务，确保 device 空闲
    npu-smi info
    
    # 执行离线推理
    rm -rf result/
    mkdir result
    python ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model "./WaveGlow_om.om" --input ./data/LJ001-0001.wav.bin --dymDims mel:1,80,832 --output "./result" --outfmt BIN --batchsize 1
    ```

- 参数说明：

  -   ${ais_infer_path}/ais_infer.py：推理脚本路径。
  -   model：om 模型路径。
  -   input：预处理后的 bin 文件存放路径。
  -   dymDims：指定推理的shape
  -   output：输出文件存放路径
  -   outfmt：输出文件格式

  推理后的输出默认在当前目录result下。

  >**说明：** 
  >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见[《ais_infer 推理工具使用文档》](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)

 3. 数据后处理

    执行WaveGlow_postprocess.py脚本对ais_infer推理结果进行后处理，得到'.wav'音频文件

    ```
    python WaveGlow_postprocess.py -f result/2022_08_26-09_40_18/LJ001-0001_0.bin -o final/1/
    ```

    **参数说明：**

    > -f 推理结果路径
    > -o 后处理结果存放路径
    > ./result/2022_xx_xx-xx_xx_xx/LJ001-0001_0.bin 中的 2022_xx_xx-xx_xx_xx 为 ais_infer 自动生成的目录名

## 模型推理性能

调用ACL接口推理计算，性能参考下列数据。

不同batch下各设备的吞吐率性能（fps）

|           | 310P3 | T4    | 310P/T4 |
| --------- | ----- | ----- | ------- |
| batch1    | 0.59  | 0.037 | 15.946  |
| batch4    | 2.34  | 0.057 | 41.053  |
| batch8    | 4.68  | 0.061 | 76.721  |
| batch16   | 9.35  | 0.076 | 123.026 |
| 最优batch | 9.35  | 0.076 | 123.026 |

最优的310P3性能达到最优的T4性能的123.026倍。