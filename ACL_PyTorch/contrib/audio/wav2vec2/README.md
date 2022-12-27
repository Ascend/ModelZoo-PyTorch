

- [概述](#概述)
  - [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [环境准备](#环境准备)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
- [模型推理性能&精度](#模型推理性能精度)

# 概述

wav2vec2 是一个用于语音表示学习的自监督学习框架，它完成了原始波形语音的潜在表示并提出了在量化语音表示中的对比任务。对于语音处理，该模型能够在未标记的数据上进行预训练而取得较好的效果。在语音识别任务中，该模型使用少量的标签数据也能达到最好的半监督学习效果。

- 参考实现：
  ```
  url=https://github.com/huggingface/transformers<br>
  commit_id=39b4aba54d349f35e2f0bd4addbe21847d037e9e<br>
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

  **表 1**  版本配套表

| 配套                      | 版本        | 环境准备指导               |
| ------------------------- | -------    | -------------------------- |
| 固件与驱动                 | 1.0.15     | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                      | 5.1.RC2    | -                          |
| Python                    | 3.7        | -                          |
| PyTorch                   | 1.10.0     | -                          |
| magiconnx                 | 0.1.0      | [MagicONNX](https://gitee.com/Ronnie_zheng/MagicONNX)                          |

# 快速上手


## 环境准备

1. 获取源码

   通过Git获取对应版本的代码并安装的方法如下：
   ```
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

3. 将[MagicONNX](https://gitee.com/Ronnie_zheng/MagicONNX) 安装到相应的Python环境

## 准备数据集

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本项目使用的数据是 [LibriSpeech](http://www.openslr.org/12) 数据集，其中测试使用的是 [test-clean](https://openslr.magicdatatech.com/resources/12/test-clean.tar.gz) 部分，因此只需要将测试数据集 test-clean 部分下载到 `/opt/npu` 目录或者其它目录并解压就可以了。

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据,执行wav2vec2_preprocess.py脚本，完成预处理。

   **预处理脚本**
   ```
   python3 wav2vec2_preprocess.py --input=${dataset_path} --batch_size=16 --output="./data/bin_om_input/bs16/"
   ```
   - 参数说明:
     - --input: 测试数据集解压后的文件夹，这里即是 LibriSpeech 测试数据集解压后的路径<br>
     - --ouput: 测试数据集处理后保存bin文件的文件夹<br>
     - --batch_size: 数据集需要处理成的批大小<br>

   **预处理成功后生成的文件**

   - `data/bin_om_input/bs16`: 输出文件夹，预处理后bin文件存放的目录，且一批数据处理成一个bin文件，作为OM模型的输入<br>
   - `data/batch_i_filename_map_bs16.json`: 存放元数据，保存着每个批里面包含了哪些文件，即一个batch对应一个（batch为1）或多个（batch不为1）bin文件名<br>
   - `data/ground_truth_texts.txt`: 存放每个样本的真实文本数据，即一段语音文件的文件名对应的真实文本数据<br>

## 模型推理

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型.om文件。

   1. 获取 Pytorch 模型配置文件和数据集词典文件等, 可通过 `wget` 或其它方式下载，以下以 `wget` 为例：
      ```
      mkdir wav2vec2_pytorch_model && cd wav2vec2_pytorch_model

      wget https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/config.json
      wget https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/feature_extractor_config.json
      wget https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/preprocessor_config.json
      wget https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/special_tokens_map.json
      wget https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/tokenizer_config.json
      wget https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/vocab.json

      ```

   2. 获取模型权重文件。

      [模型权重文件](https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/pytorch_model.bin)使用官方开源的pyTorch模型， 必须将该权重文件下载到`wav2vec2_pytorch_model` 目录下和模型配置文件、词典文件等放在一起，然后回到主目录执行其它脚本。
      ```
      wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/pytorch_model.bin
      cd ..
      ```

   3. 导出 ONNX 模型文件。

      使用 `wav2vec2_path2onnx` 导出 ONNX 模型文件。

      ```shell
      python3 wav2vec2_pth2onnx.py --pytorch_model_dir="./wav2vec2_pytorch_model" --output_model_path="./wav2vec2.onnx"
      ```

      - 参数说明：
         - --pytorch_model_dir：Pytoch模型及其配置文件的目录。
         - --output_model_path：导出ONNX模型的路径。

   3. ONNX 模型优化，模型优化的[参考思路](https://gitee.com/shikang2022/docs-openmind/blob/master/guide/modelzoo/onnx_model/tutorials/%E4%B8%93%E9%A2%98%E6%A1%88%E4%BE%8B/%E6%80%A7%E8%83%BD%E8%B0%83%E4%BC%98/%E6%A1%88%E4%BE%8B-Conv1D%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96.md)

      ```
      python3 wav2vec2_modify.py --input_model_path="wav2vec2.onnx" --output_model_path="wav2vec2_modified.onnx"
      ```

      - 参数说明：
         - --input_model_path：需要优化的 ONNX 模型路径。
         - --output_model_path：优化完 ONNX 模型保存的路径。

   4. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：**
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

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

      3. 执行 [ATC](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/51RC2alpha003/infacldevg/atctool/atlasatc_16_0007.html) 命令。

         执行 ATC 命令：

         ```
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
           -   --op_precision_mode：自定义算子精度模式
           -   --log：日志级别。`
           -   --soc\_version：处理器型号。

           运行成功后生成 **wav2vec2_bs${bs}.om** 模型文件。


2. 开始推理验证。

   根据OS架构选择的推理工具，执行命令增加工具可执行权限，在此以ais_bench推理工具为例。

   a. 安装ais_bench推理工具

      ais_bench推理工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310P上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考。   
	  请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

   b. 执行推理。

      执行推理之前应先创建保存推理结果的文件，否则无法保存：
      ```
      mkdir ./data/bin_om_output
      ```
      执行推理脚本:
      ```
      python3 -m ais_bench --model "wav2vec2_bs${bs}.om" \
                           --input "./data/bin_om_input/bs${bs}/" \
                           --output "./data/bin_om_output" \
                           --batchsize ${bs}
      ```

      - 参数说明：
         - --model：OM模型文件。
         - --input：前处理得到的bin数据文件目录。
         - --ouput：推理出结果的bin文件保存的目录
         - --batchsize: 批大小

   c. 执行后处理脚本并进行精度验证

      ```
      python3 wav2vec2_postprocess.py --input "./data/bin_om_output/${timestamp}" --batch_size ${bs}
      ```
      - 参数说明：
        - --input：OM 模型输出的二进制文件所在的文件夹
        - --batch_size：批大小


# 模型推理性能&精度

- GPU性能

   |GPU 芯片类型 | Batch Size|性能|
   |----|----|----|
   |T4|1|105|
   |T4|4|131|
   |T4|8|132|
   |T4|16|137|
   |T4|\*32|\*138|
   |T4|64|96|

- NPU 精度和性能

   | NPU芯片型号 | Batch Size     | 数据集      |    精度    |   性能    |
   | ---------  | -------------- | ----------  | ---------- | --------------- |
   |Ascend310P3 |      1         | LibriSpeech |    2.96%  |     138         |
   |Ascend310P3 |      4         | LibriSpeech |    2.96%  |     152         |
   |Ascend310P3 |      8         | LibriSpeech |    2.96%  |     127         |
   |Ascend310P3 |      \*16        | LibriSpeech |    2.96%  |     \*157         |
   |Ascend310P3 |      32        | LibriSpeech |    2.96%  |     116         |
   |Ascend310P3 |      64        | LibriSpeech |    2.96%  |     92          |

GPU 在 Batch Size 为32时性能最佳，NPU（Ascend310P3）在 Batch Size 为16时性能最佳，两者对比：Ascend 310P3 / GPU T4 = 1.138
