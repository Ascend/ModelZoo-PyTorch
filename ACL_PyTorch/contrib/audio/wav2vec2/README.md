# wav2vec2模型-推理指导

- [wav2vec2模型-推理指导](#wav2vec2模型-推理指导)
- [概述](#概述)
  - [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [环境准备](#环境准备)
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
  | input    | FP32 | batchsize x 10000 | ND         |


- 输出数据

  | 输出数据 | 数据类型     | 大小 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FP32 | batchsize x 312 x 32 | ND        |


# 推理环境准备

- 该模型需要以下插件与驱动


| 配套                      | 版本        | 环境准备指导               |
| ------------------------- | -------    | -------------------------- |
| 固件与驱动                 | 22.0.4.b010     | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                      | 6.0.RC1    | -                          |
| Python                    | 3.7        | -                          |
| PyTorch                   | 1.10.0     | -                          |

# 快速上手


## 环境准备

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
   ```
   pip3 install -r requirements.txt
   ```
   > 如果报错： `OSError: sndfile library not found`， 则需要执行此命令： `sudo apt-get install libsndfile1`

3. 安装改图工具 auto-optimizer
   ```bash
   git clone https://gitee.com/ascend/msadvisor.git
   cd msadvisor/auto-optimizer
   pip3 install -r requirements.txt
   python3 setup.py install
   cd ../..
   ```

## 准备数据集

1. 获取原始数据集

   本项目使用的数据是 [LibriSpeech](http://www.openslr.org/12) 数据集，其中测试使用的是 [test-clean](https://openslr.magicdatatech.com/resources/12/test-clean.tar.gz) 部分。

   ```bash
   wget https://openslr.magicdatatech.com/resources/12/test-clean.tar.gz

   # 解压到当前目录
   tar -xzvf test-clean.tar.gz
   ```

2. 数据预处理

   将原始数据集转换为模型输入的数据,执行```wav2vec2_preprocess.py```脚本，完成预处理。

   ```bash
   # 设置Batch size，请按需修改
   bs=16

   python3 wav2vec2_preprocess.py --input=${dataset_path} --batch_size=${bs} --output="./data/bin_om_input/bs${bs}/"
   ```
   - 参数说明:
     - --input: 测试数据集解压后的文件夹，这里即是 LibriSpeech 测试数据集解压后的路径。
     - --ouput: 测试数据集处理后保存bin文件的文件夹。
     - --batch_size: 数据集需要处理成的批次大小。

   预处理成功后生成的文件:

   - `data/bin_om_input/bs16`: 输出文件夹，预处理后bin文件存放的目录，且一批数据处理成一个bin文件，作为OM模型的输入
   - `data/batch_i_filename_map_bs16.json`: 存放元数据，保存着每个批里面包含了哪些文件，即一个batch对应一个（batch为1）或多个（batch不为1）bin文件名
   - `data/ground_truth_texts.txt`: 存放每个样本的真实文本数据，即一段语音文件的文件名对应的真实文本数据

## 模型推理

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型.om文件。

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

   3. 导出 ONNX 模型文件。

      使用 `wav2vec2_path2onnx` 导出 ONNX 模型文件。

      ```bash
      python3 wav2vec2_pth2onnx.py --pytorch_model_dir="./wav2vec2_pytorch_model" --output_model_path="./wav2vec2.onnx"
      ```

      - 参数说明：
         - --pytorch_model_dir：Pytoch模型及其配置文件的目录。
         - --output_model_path：导出ONNX模型的路径。

   3. ONNX 模型优化

      ```bash
      python3 wav2vec2_modify.py --input_model_path="wav2vec2.onnx" --output_model_path="wav2vec2_modified.onnx"
      ```

      - 参数说明：
         - --input_model_path：需要优化的 ONNX 模型路径。
         - --output_model_path：优化完 ONNX 模型保存的路径。

   4. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```bash
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：**
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```bash
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

      3. 执行 ATC 命令：

         ```bash
         # 设置Batch size，请按需修改
         bs=16

         atc --model=wav2vec2_modified.onnx \
             --framework=5 \
             --input_shape="modelInput:${bs},100000" \
             --output=wav2vec2_bs${bs} \
             --op_precision_mode=op_precision.ini \
             --soc_version=${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --op_precision_mode：自定义算子精度模式。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成 ```wav2vec2_bs${bs}.om``` 模型文件。


2. 开始推理验证。

   在此以ais_bench推理工具为例。

   a. 安装ais_bench推理工具     
	  请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   b. 执行推理

      执行推理脚本:
      ```bash
      # 创建保存推理结果的父目录
      mkdir ./data/bin_om_output

      python3 -m ais_bench --model "wav2vec2_bs${bs}.om" \
                           --input "./data/bin_om_input/bs${bs}/" \
                           --output "./data/bin_om_output" \
                           --output_dirname "bs${bs}"
      ```

      - 参数说明：
         - --model：OM模型文件
         - --input：前处理得到的bin数据文件目录
         - --ouput：推理结果父目录
         - --output_dirname：保存推理结果bin文件的子目录

   c. 执行后处理脚本并进行精度验证

      ```bash
      python3 wav2vec2_postprocess.py --input "./data/bin_om_output/bs${bs}" --batch_size ${bs}
      ```
      - 参数说明：
        - --input：OM 模型输出的二进制文件所在的文件夹
        - --batch_size：批次大小

   d. 执行纯推理验证性能
      ```bash
      python3 -m ais_bench --model "wav2vec2_bs${bs}.om" --loop 100
      ```

      - 参数说明：
         - --model：OM模型文件
         - --loop：纯推理次数


# 模型推理性能&精度

   | NPU芯片型号 | Batch Size |    数据集    |  精度    |  性能  | 基准性能 |
   | :-------:  | :--------: | :---------: | :-----: | :----: | :----: |
   |Ascend310P3 |      1     | LibriSpeech |  0.970  |134.0863|132.90|
   |Ascend310P3 |      4     | LibriSpeech |  0.970  |143.9766|124.09|
   |Ascend310P3 |      8     | LibriSpeech |  0.970  |142.2380|133.91|
   |Ascend310P3 |      16    | LibriSpeech |  0.970  |144.3543|127.42|
   |Ascend310P3 |      32    | LibriSpeech |  0.970  |131.5986|118.67|
   |Ascend310P3 |      64    | LibriSpeech |  0.970  |118.9236|119.69|


