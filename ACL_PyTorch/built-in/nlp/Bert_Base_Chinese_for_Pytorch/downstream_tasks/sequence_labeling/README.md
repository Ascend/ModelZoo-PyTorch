# Bert_Base_Chinese模型下游任务-Sequence_Labeling-推理指导

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

`sequence_labeling`: 基于BERT+CRF的序列标注子任务

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

  | 输入数据  | 数据类型 | 大小                      | 数据排布格式 |
  | --------  | -------- | ------------------------- | ------------ |
  | token_ids | INT64    | batchsize x seq_len       | ND           |

- 输出数据

  | 输出数据       | 大小     | 数据类型                                 | 数据排布格式 |
  | --------       | -------- | --------                                 | ------------ |
  | emission_score | FP16     | batchsize x seq_length x length_category | ND           |
  | attention_mask | FP16     | batchsize x seq_length                   | ND           |

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
   cd ACL_PyTorch/built-in/nlp/Bert_Base_Chinese_for_Pytorch/downstream_tasks/sequence_labeling              # 切换到模型的代码仓目录
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
1. 获取原始数据集。

   下载china-people-daily数据：

   ```
   wget https://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
   ```

   解压得到数据文件：

   ```
   tar -xf china-people-daily-ner-corpus.tar.gz
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行preprocess.py脚本，完成预处理。

   ```shell
   python3 preprocess.py --input_path ./china-people-daily-ner-corpus/example.dev --out_dir ./preprocessed_data --dict_path ./bert-base-chinese/vocab.txt
   ```

   - 参数说明：
     - --input_path：输入数据集文件路径。
     - --out_dir：预处理生成数据所在路径。
     - --dict_path：预处理数据所需模型数据配置文件。

### 模型推理<a name="section741711594517"></a>

1. 模型转换

   1. 获取权重文件

     获取权重文件：[best_model.pt](https://pan.baidu.com/s/1-cQ3hpB-SmB94NqwO5_7Dw), 提取码：rasv。

   2. 导出onnx文件

     ```shell
     python3 pth2onnx.py --input_path best_model.pt --out_path ./models/onnx/bert_base_chinese_sequence_labeling.onnx --config_path ./bert-base-chinese/config.json
     ```
 
     - 参数说明：
       - --input_path：输入模型权重路径。
       - --out_path：输出onnx模型所在路径。
       - --config_path：模型配置文件。

   3. 模型优化

     通过onnx-simplifier等对onnx进行优化：

     ```shell
     # 修改优化模型：以bs64为例
     python3 -m onnxsim ./models/onnx/bert_base_chinese_sequence_labeling.onnx ./models/onnx/bert_base_chinese_bs64.onnx --input-shape "token_ids:64,256"
     python3 fix_onnx.py ./models/onnx/bert_base_chinese_bs64.onnx ./models/onnx/bert_base_chinese_bs64_fix.onnx
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
         ```shell
         mkdir -p models/om
         # bs:[1, 4, 8, 16, 32, 64]
         atc --model=./models/onnx/bert_base_chinese_bs${bs}_fix.onnx --framework=5 --output=./models/om/bert_base_chinese_bs${bs} --input_format=ND --log=debug --soc_version=${chip_name} --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance
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

   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

        ```
        # 以bs64模型推理为例
        mkdir -p ./output_data/bs64
        python3 -m ais_bench --model ./models/om/bert_base_chinese_bs64.om --input ./preprocessed_data/input_data --output ./output_data --output_dirname bs64 --batchsize 64 --device 0
        ```
        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入文件。
             -   --output：输出目录。
             -   --output_dirname：输出文件名。
             -   --device：NPU设备编号。
             -   --outfmt: 输出数据格式。
             -   --batchsize：推理模型对应的batchsize。


        推理后的输出默认在当前目录output_data/bs64下。

   3.  精度验证。

      调用postprocess.py脚本与数据集标签比对，获得Accuracy数据。

      ```
      # 以bs64模型推理为例
      python3 postprocess.py --result_dir output_data/bs64 --out_path eval.json --label_dir preprocessed_data/label --config_path ./bert-base-chinese/config.json --ckpt_path ./best_model.pt
      ```
      - 参数说明：

           -   --result_dir：推理结果所在路径。
           -   --out_path：输出结果所在路径。
           -   --label_dir：GT label所在路径。
           -   --config_path：模型配置文件。
           -   --ckpt_path：模型权重文件路径。


## 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

精度：

| 模型              | Pth精度(val-token level)                   | Pth精度(val-entity level)                  |
| :---------------: | :--------:                                 | :-------------:                            |
| Bert-Base-Chinese | f1:0.9724 precision: 0.9684 recall: 0.9765 | f1:0.9600 precision: 0.9569 recall: 0.9632 |
| 模型              | NPU精度(val-token level)                   | NPU精度(val-entity level)                  |
| Bert-Base-Chinese | f1:0.9723 precision: 0.9684 recall: 0.9763 | f1:0.9603 precision: 0.9571 recall: 0.9635 |

性能：

| 模型              | BatchSize | 310P性能 |  基准性能 |
| :---------------: | :-------: |  :-----: | :-------: |
| Bert-Base-Chinese |         1 |      299 |       701 |
| Bert-Base-Chinese |         4 |      539 |      1227 |
| Bert-Base-Chinese |         8 |      592 |      1360 |
| Bert-Base-Chinese |        16 |      572 |      1429 |
| Bert-Base-Chinese |        32 |      564 |      1480 |
| Bert-Base-Chinese |        64 |      545 |      1468 |
