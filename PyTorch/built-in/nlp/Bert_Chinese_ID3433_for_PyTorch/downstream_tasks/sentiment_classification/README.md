# Bert_Base_Chinese模型下游任务-Sentiment_Classification-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

## 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

`BERT`来自 Google 的论文`Pre-training of Deep Bidirectional Transformers for Language Understanding`，`BERT` 是`Bidirectional Encoder Representations from Transformers`的首字母缩写，整体是一个自编码语言模型。`Bert_Base_Chinese`是`BERT`模型在中文语料上训练得到的模型。

`sentiment_classification`: 基于BERT的情感文本分类子任务

  预训练模型：
  ```shell
  url=https://huggingface.co/bert-base-chinese
  commit_id=38fda776740d17609554e879e3ac7b9837bdb5ee
  mode_name=Bert_Base_Chinese
  ```

  依赖仓：
  ```shell
  url=https://github.com/Tongjilibo/bert4torch.git
  commit_id=c348349a4c7579d14c393ea61e868652801293ca
  ```

### 输入输出数据<a name="section540883920406"></a>

- 输入数据

    |  输入数据 |  大小 |  数据类型 |  数据排布格式 |
    |---|---|---|---|
    |  token_ids | FP16  | batchsize x seq_length  | ND  |
    |  segment_ids |  FP16 | batchsize x seq_length  | ND  |



- 输出数据

  | 输出数据       | 大小     | 数据类型                                 | 数据排布格式 |
  | --------       | -------- | --------                                 | ------------ |
  | output| FP16     | batchsize x 2 | ND           |

## 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                            | 版本    | 环境准备指导                                                                                          |
| ------------------------------------------------------------    | ------- | ------------------------------------------------------------                                          |
| 固件与驱动                                                      | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                            | 6.0.RC1 | -                                                                                                     |
| Python                                                          | 3.7.5   | -                                                                                                     |
| PyTorch                                                         | 1.5.0+ | -                                                                                                     |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |

## 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

### 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git        # 克隆仓库的代码
   git checkout master         # 切换到对应分支
   cd ACL_PyTorch/built-in/nlp/Bert_Base_Chinese_for_Pytorch/downstream_tasks/sentiment_classification              # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```shell
   pip3 install -r requirement.txt
   # 改图
   git clone https://gitee.com/Ronnie_zheng/MagicONNX.git MagicONNX
   cd MagicONNX && git checkout dev
   pip3 install . && cd ..
   ```

2. 获取开源代码仓。
   在已下载的源码包根目录下，执行如下命令。

   ```shell
   # 模型依赖
   git clone https://github.com/Tongjilibo/bert4torch.git
   cd bert4torch && git checkout c348349a4c7579d14c393ea61e868652801293ca
   pip3 install . && cd ..
   git clone https://huggingface.co/bert-base-chinese
   ```

### 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集，原始模型权重以及词表。

   下载sentiment数据：

   ```
   wget https://github.com/bojone/bert4keras/blob/master/examples/datasets/sentiment.zip
   ```

   解压得到数据文件：

   ```
   unzip sentiment.zip
   ```
   
   下载bert_chinese权重和词表文件
   ```
   wget https//storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
   ```
   解压得到数据文件：

   ```
   unzip chinese_L-12_H-768_A-12.zip
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行preprocess.py脚本，完成预处理。

   ```shell
   python3 preprocess.py --input_path ./sentiment/sentiment.valid.data --out_dir ./preprocessed_data --dict_path ./chinese_L-12_H-768_A-12/vocab.txt
   ```

   - 参数说明：
     - --input_path：输入数据集文件路径。
     - --out_dir：预处理生成数据所在路径。
     - --dict_path：预处理数据所需模型数据配置文件。

### 模型推理<a name="section741711594517"></a>

1. 模型转换

   1. 获取权重文件
     [权重链接](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Bert-Base_Chinese/PTH/Sentiment_Cls/best_model.pt)

   2. 导出onnx文件

     ```shell
     python3 pth2onnx.py --input_path best_model.pt --out_path ./models/onnx/bert_sentiment.onnx --config_path ./chinese_L-12_H-768_A-12/bert_config.json
     ```
 
     - 参数说明：
       - --input_path：输入模型权重路径。
       - --out_path：输出onnx模型所在路径。
       - --config_path：模型配置文件。

   3. 模型优化

     通过onnx-simplifier等对onnx进行优化：

     ```shell
     # 修改优化模型：以bs64为例
     python3 -m onnxsim ./models/onnx/bert_sentiment.onnx ./models/onnx/bert_sentiment_bs64.onnx --input-shape "token_ids:64,256" "segment_ids:64,256"
     python3 fix_onnx.py ./models/onnx/bert_sentiment_bs64.onnx ./models/onnx/bert_sentiment_bs64_fix.onnx
     ```

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
         +-------------------|-----------------|------------------------------------------------------+
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
        以64batch size为例
         ```shell
         mkdir -p models/om
         atc --model=./models/onnx/bert_sentiment_bs64_fix.onnx --framework=5 --output=./models/om/bert_sentiment_bs64 --input_format=ND --log=debug --soc_version=${chip_name} --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成bert_base_chinese_bs${bs}.om模型文件。



2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        # 以bs64模型推理为例
        mkdir -p ./output_data/bs64
        python3 -m ais_bench --model ./models/om/bert_base_chinese_bs64.om --input ./preprocessed_data/input_data --output ./output_data --output_dirname sentiment_output --batchsize 64 --device 0
        ```
        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入文件。
             -   --output：输出目录。
             -   --output_dirname：输出文件名。
             -   --device：NPU设备编号。
             -   --outfmt: 输出数据格式。
             -   --batchsize：推理模型对应的batchsize。


        推理后的输出在当前目录output_data/sentiment_output下。

   3.  精度验证。

      调用postprocess.py脚本与数据集标签比对，获得Pth模型的Accuracy数据。

      ```
      # 以bs1模型推理为例
      python3 postprocess.py -i ./sentiment/sentiment.valid.data -o eval.json -c ./chinese_L-12_H-768_A-12/bert_config.json -k ./best_model.pt -d ./chinese_L-12_H-768_A-12/vocab.txt -b 1
      ```
      - 参数说明：

           -   -i：输入数据文件
           -   -o：输出结果所在路径。
           -   -c：模型配置文件。
           -   -k：模型权重文件路径。
           -   -d：处理输入数据所需模型数据配置文件。
           -   -b：batch size
     
    调用get_om_result.py脚本与数据集标签比对，获得om模型在npu上的Accuracy数据。
    ```
      # 以bs1模型推理为例
      python3 get_om_result.py ./output/sentiment_output/ ./output/label/
      ```
     - 参数说明：
            第一个参数是之前调用ais_bench工具获得的模型输出文件路径，第二个参数是标签文件路径。
       


## 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

精度：

| 模型  | Pth精度  |
|---|---|
|  bert_sentiment |f1: 0.9502 precision: 0.9502 recall 0.9503 |
|  模型 | NPU精度  |
|  bert_sentiment |f1: 0.9502 precision: 0.9502 recall 0.9502   |


性能：

| 模型              | BatchSize | 310P性能 |  基准性能 |
| :---------------: | :-------: |  :-----: | :-------: |
| bert_sentiment  |         1 |      323 |       758 |
| bert_sentiment  |         4 |      558 |      1284 |
| bert_sentiment  |         8 |      580|      1424|
| bert_sentiment  |        16 |      554|      1488|
| bert_sentiment  |        32 |      550|      1568|
| bert_sentiment  |        64 |      546|      1480|
