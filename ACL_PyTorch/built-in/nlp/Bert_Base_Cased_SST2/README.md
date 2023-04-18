# Bert_Base_Cased模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

- [其他下游任务](#ZH-CN_TOPIC_0000001126121892)


## 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

`BERT`来自 Google 的论文`Pre-training of Deep Bidirectional Transformers for Language Understanding`，`BERT` 是`Bidirectional Encoder Representations from Transformers`的首字母缩写，整体是一个自编码语言模型。`Bert_Base_Cased`是`BERT`模型在Glue上训练得到的模型。本模型以Glue下的SST-2任务为例。

- 参考实现：

  ```
  url=https://huggingface.co/gchhablani/bert-base-cased-finetuned-sst2
  commit_id=e3a2a13efbaaf56afd02eb7333952ea22a693c45
  ```

### 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据       | 数据类型 | 大小                      | 数据排布格式 |
  | --------       | -------- | ------------------------- | ------------ |
  | input_ids      | INT64    | batchsize x seq_len       | ND           |
  | attention_mask | INT64    | batchsize x seq_len       | ND           |

- 输出数据

  | 输出数据 | 数据类型 | 大小   | 数据排布格式 |
  | -------- | -------- | ------------- | ------------ |
  | output   |  FLOAT32 |batch_size x 2 | ND           |


## 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                            | 版本    | 环境准备指导                                                                                          |
| ------------------------------------------------------------   | ------- | ------------------------------------------------------------                                          |
| 固件与驱动                                                      | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                            | 6.0.RC1 | -                                                                                                     |
| Python                                                          | 3.7.5   | -                                                                                                     |
| PyTorch                                                         | 1.8.0 | -                                                                                                     |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |

## 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

### 获取源码<a name="section4622531142816"></a>

1. 获取源码。
   ```shell
   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/gchhablani/bert-base-cased-finetuned-sst2
   ```

2. 安装依赖。

   ```shell
   pip3 install -r requirements.txt
   git clone https://gitee.com/Ronnie_zheng/MagicONNX.git MagicONNX
   cd MagicONNX && git checkout dev
   pip3 install . && cd ..
   ```


### 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。
   ```
   mkdir data
   ```
   本模型采用SST-2数据集的验证集进行精度评估。获取[SST-2官方数据集](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip)并解压后将SST-2文件夹放在data文件夹下。目录结构如下：

   ```
   Bert_Base_Cased_SST2
   ├── data
      ├──SST-2
            ├── test.tsv
            ├── dev.tsv
            ├── train.tsv
            └── original
   ```


2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行preprocess.py脚本，完成预处理。

   ```shell
   python3 preprocess.py \
            --text_file data/SST-2/dev.tsv \
            --tokenizer_path bert-base-cased-finetuned-sst2 \
            --seq_len ${seq_len} \
            --save_path data_bin
   ```

   - 参数说明：
      - --text_file：验证集文件路径。
      - --tokenizer_path：配置路径。
      - --save_path：预处理数据路径
      - --sqe_len：sequence长度。

### 模型推理<a name="section741711594517"></a>

1. 模型转换

   1. 获取权重文件

      获取权重文件：[pytorch_model.bin](https://huggingface.co/gchhablani/bert-base-cased-finetuned-sst2/resolve/main/checkpoint-12630/pytorch_model.bin)，替换`bert-base-cased-finetuned-sst2/checkpoint-12630`目录下的文件：

      ```shell
      mv pytorch_model.bin bert-base-cased-finetuned-sst2/checkpoint-12630/
      ```

   2. 导出onnx文件

      ```shell
      python3 pth2onnx.py \
              --model_dir bert-base-cased-finetuned-sst2/checkpoint-12630/ \
              --save_path ./bert_base_sst2.onnx
      ```
      运行成功后得到`bert_base_sst2.onnx`模型文件
      
      - 参数说明：
         - --model_dir：模型路径（包含配置文件等）
         - --save_path：输出的onnx文件路径。

   3. 优化onnx文件

      ```shell
      python3 -m onnxsim ./bert_base_sst2.onnx ./bert_base_sst2_bs${bs}.onnx --input-shape "input_ids:${bs},${seq_len}" "attention_mask:${bs},${seq_len}"
      python3 modify_onnx.py bert_base_sst2_bs${bs}.onnx bert_base_sst2_bs${bs}_md.onnx
      ```
      运行成功后得到`bert_base_sst2_bs${bs}_md.onnx`模型文件

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
         atc --model=./bert_base_sst2_bs${bs}_md.onnx \
             --framework=5 \
             --output=./bert_base_sst2_bs${bs} \
             --input_format=ND \
             --log=error \
             --soc_version=Ascend${chip_name} \
             --optypelist_for_implmode="Gelu" \
             --op_select_implmode=high_performance
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成`bert_base_sst2_bs${bs}.om`模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      ```shell
      python3 -m ais_bench \
               --model ./bert_base_sst2_bs${bs}.om \
               --input ./data_bin/input_ids,./data_bin/attention_mask \
               --output ./data_bin/ \
               --output_dirname output
      ```
      -   参数说明：

            -   --model：om文件路径。
            -   --input：输入文件。
            -   --output：输出目录。
            -   --output_dirname：输出子目录。


        推理后的输出默认在当前目录data_bin/output下。

   3. 精度验证。

      调用postprocess.py脚本与数据集标签比对，获得Accuracy数据。

      ```shell
      python3 postprocess.py \
               --output_path ./data_bin/output \
               --label_path ./data_bin/labels.json
      ```
      
      - 参数说明：
         - --output_path：推理结果路径。
         - --label_path：label文件路径。

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```shell
      python3 -m ais_bench --model=./bert_base_sst2_bs${bs}.om --loop=100
      ```

      - 参数说明：
        - --model：om模型文件路径。
        - --loop：推理次数。

## 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>
调用ACL接口推理计算，性能参考下列数据。

seq_len = 128
| 芯片型号 | Batch Size   |  数据集  |   精度   |   性能   |
| -------- | --------- | ---------- | -------- | ----------- |
|   310P   |    1      |   SST-2    |   92.43  |   483.08     |
|   310P   |    4      |   SST-2    |   92.43  |   1136.52    |
|   310P   |    8      |   SST-2    |   92.43  |   1336.38    |
|   310P   |    16     |   SST-2    |   92.43  |   1379.32    |
|   310P   |    32     |   SST-2    |   92.43  |   1254.83    |
|   310P   |    64     |   SST-2    |   92.43  |   1220.88    |
