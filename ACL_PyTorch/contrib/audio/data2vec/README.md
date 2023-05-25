# Data2vec模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

[comment]: <> (- [配套环境]&#40;#ZH-CN_TOPIC_0000001126121892&#41;)

  ******





# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>
Data2Vec是一个对语音、语言或计算机视觉使用相同学习方法的框架，核心思想是使用标准的Transformer 架构，在self-distillation（自蒸馏）设置中基于输入的掩码视图来预测完整输入数据的潜在表示。针对语音部分，编码方式上采用一维卷积网络，掩码方式与wav2vec一致。



- 参考实现：

  ```
  url=https://github.com/huggingface/transformers/tree/main/src/transformers/models/data2vec
  commit_id= a6937898c117a2f75c3ee354eb2e4916f428f441
  model_name= modeling_data2vec_audio
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FLOAT32 | batchsize x 559280 | ND         |


- 输出数据

  | 输出数据 | 数据类型     | 大小 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32| batchsize x 1747 x 32  | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

   | 配套                                                         | 版本    | 环境准备指导                                                 |
   | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
   | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
   | CANN                                                         | 6.0.RC1 | -                                                            |
   | Python                                                       | 3.7.13   | -                                                            |
   | PyTorch                                                      | 1.10.0   | -                                                            |
   | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>
1. 安装依赖。

   ```
   pip3 install -r requirement.txt

   ```
   > 如果报错： `OSError: sndfile library not found`， 则需要执行此命令： `sudo apt-get install libsndfile1`


## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
   
   本模型使用LibriSpeech测试集，下载 [LibriSpeech-test_clean](https://www.openslr.org/resources/12/test-clean.tar.gz) 后解压。目录结构如下：
   ```
   data2vec
   ├── transformers
   ├── LibriSpeech
      ├── test-clean
      ├── BOOKS.TXT
      ├── ...
   ```



2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行data2vec_preprocess.py脚本，完成预处理。

   ```
   python3 data2vec_preprocess.py --input="./LibriSpeech/test-clean/" --output="./data/bin_in"
   ```
   运行后生成的文件存放在/data/bin_in


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       
       ```
        mkdir data2vec_pytorch_model
        wget https://huggingface.co/facebook/data2vec-audio-base-960h/raw/main/README.md -P data2vec_pytorch_model
        wget https://huggingface.co/facebook/data2vec-audio-base-960h/raw/main/config.json -P data2vec_pytorch_model
        wget https://huggingface.co/facebook/data2vec-audio-base-960h/raw/main/preprocessor_config.json -P data2vec_pytorch_model
        wget https://huggingface.co/facebook/data2vec-audio-base-960h/resolve/main/pytorch_model.bin -P data2vec_pytorch_model
        wget https://huggingface.co/facebook/data2vec-audio-base-960h/raw/main/special_tokens_map.json -P data2vec_pytorch_model
        wget https://huggingface.co/facebook/data2vec-audio-base-960h/raw/main/tokenizer_config.json -P data2vec_pytorch_model
        wget https://huggingface.co/facebook/data2vec-audio-base-960h/raw/main/vocab.json -P data2vec_pytorch_model
       ```

   2. 导出onnx文件。
   
      1. 使用data2vec_pth2onnx.py导出onnx文件。

         运行data2vec_pth2onnx.py脚本。

         ```
         python3 data2vec_pth2onnx.py
         ```

         获得data2vec.onnx文件。
      
      2. 优化ONNX文件。(安装[auto-optimzer](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer)工具)

         ```
         python3 data2vec_modify.py -m1 data2vec.onnx -m2 data2vec_new.onnx
         ```
         - 参数说明：
            - --input_name(m1)：onnx文件路径。
            - --output_name(m2): 优化后的onnx文件路径。

         获得data2vec_new.onnx文件
    

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

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

      3. 执行ATC命令。

         ```
         atc --model=data2vec_new.onnx \
             --framework=5 \
             --input_shape="modelInput:${bs},559280" \
             --output=data2vec_bs${bs} \
             --soc_version=Ascend${chip_name} \
             --op_precision_mode=op_precision.ini
         ```

         - 参数说明：

           - --model：为ONNX模型文件。
           - --framework：5代表ONNX模型。
           - --output：输出的OM模型。
           - --input\_shape：输入数据的shape。
           - --soc\_version：处理器型号。 
           - --op_precision_mode：设置算子精度模式。
           
         运行成功后生成data2vec_bs${bs}.om模型文件。



2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2.  执行推理。

      ```
      python3 -m ais_bench \
              --model data2vec_bs${bs}.om \
              --input "data/bin_in" \
              --output "./" \
              --output_dirname "data/bin_out"
      ```

      -   参数说明：\
         - --model：om文件路径。
         - --input：输入数据所在的文件夹。
         - --output：推理结果输出路径。
         - --output_dirname：推理结果输出子文件夹。

      推理后的输出默认在data/bin_out目录下。


   3.  数据后处理
      调用数据后处理脚本将模型输出转换为文本。

      ```
      python3 data2vec_postprocess.py --input "data/bin_out"
      ```
      运行成功后得到文件/data/infered_texts.txt。

   4.  精度验证。

      调用脚本与数据集标签ground_truth_texts.txt比对，可以获得Accuracy数据。

      ```
      python3 data2vec_eval_accuracy.py --ground_truth_text "./data/ground_truth_texts.txt" --infered_text "./data/infered_texts.txt"
      ```
      -   参数说明\
         - --ground_truth_text：真实文本所在的路径。
         - --infered_text：推理文本所在的路径。

    
    

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

   | NPU芯片型号 | Batch Size     | 数据集      | 精度(wer)  | 性能 (fps)      |
   | ---------  | -------------- | ----------  | ---------- | --------------- |
   |Ascend310P3 |      1         | LibriSpeech |    0.938  |     11.154       |
   |Ascend310P3 |      4         | LibriSpeech |    0.938  |     10.351       |
   |Ascend310P3 |      8         | LibriSpeech |    0.938  |     10.391       |
   |Ascend310P3 |      16        | LibriSpeech |    0.938  |     10.561       |
   |Ascend310P3 |      32        | LibriSpeech |    0.938  |     10.561       |
   |Ascend310P3 |      64        | LibriSpeech |    0.938  |     10.292       |


