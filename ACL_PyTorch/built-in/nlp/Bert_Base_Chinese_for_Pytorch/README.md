# Bert_Base_Chinese模型-推理指导

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

`BERT`来自 Google 的论文`Pre-training of Deep Bidirectional Transformers for Language Understanding`，`BERT` 是`Bidirectional Encoder Representations from Transformers`的首字母缩写，整体是一个自编码语言模型。`Bert_Base_Chinese`是`BERT`模型在中文语料上训练得到的模型。

  ```shell
  url=https://huggingface.co/bert-base-chinese
  commit_id=38fda776740d17609554e879e3ac7b9837bdb5ee
  mode_name=Bert_Base_Chinese
  ```

### 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据       | 数据类型 | 大小                      | 数据排布格式 |
  | --------       | -------- | ------------------------- | ------------ |
  | input_ids      | INT64    | batchsize x seq_len       | ND           |
  | attention_mask | INT64    | batchsize x seq_len       | ND           |
  | token_type_ids | INT64    | batchsize x seq_len       | ND           |

- 输出数据

  | 输出数据 | 大小               | 数据类型 | 数据排布格式 |
  | -------- | --------           | -------- | ------------ |
  | output   | batch_size x class | FLOAT32  | ND           |


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
   cd ACL_PyTorch/built-in/nlp/Bert_Base_Chinese_for_Pytorch              # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   git clone https://gitee.com/Ronnie_zheng/MagicONNX.git MagicONNX
   cd MagicONNX && git checkout dev
   pip3 install . && cd ..
   ```

2. 获取开源代码仓。
   在已下载的源码包根目录下，执行如下命令。

   ```shell
   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/bert-base-chinese
   ```

### 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。

   如果你想重新处理zhwiki的原始数据，可按照以下步骤操作。

   下载zhwiki原始数据：

   ```
   wget https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2 --no-check-certificate
   ```

   解压得到zhwiki-latest-pages-articles.xml

   ```
   bzip2 -dk zhwiki-latest-pages-articles.xml.bz2
   ```

   下载预处理脚本：

   ```shell
    wget https://github.com/natasha/corus/raw/master/corus/third/WikiExtractor.py
   ```

   使用WikiExtractor.py提取文本，其中extracted/wiki_zh为保存路径，建议不要修改：

   ```
   python3 WikiExtractor.py zhwiki-latest-pages-articles.xml -b 100M -o extracted/wiki_zh
   ```

   将多个文档整合为一个txt文件，在本工程根目录下执行

   ```
   python3 WikicorpusTextFormatting.py --extracted_files_path extracted/wiki_zh --output_file zhwiki-latest-pages-articles.txt
   ```

   最终生成的文件名为zhwiki-latest-pages-articles.txt (也可直接采用处理好的文件)

   从中分离出验证集：

   ```shell
   python3 split_dataset.py zhwiki-latest-pages-articles.txt zhwiki-latest-pages-articles_validation.txt
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行preprocess.py脚本，完成预处理。

   ```shell
   # 输入参数：${input_patgh} ${model_dir} ${save_dir} ${seq_length}
   python3 preprocess.py ./zhwiki-latest-pages-articles_validation.txt ./bert-base-chinese ./input_data/ 384
   ```

   - 参数说明：第一个参数为zhwiki数据集分割得到验证集文件路径，第二个参数为源码路径（包含模型配置文件等），第三个参数为输出预处理数据路径，第四个参数为sequence长度。

### 模型推理<a name="section741711594517"></a>

1. 模型转换

   1. 获取权重文件

      获取权重文件：[pytorch_model.bin](https://huggingface.co/bert-base-chinese/blob/main/pytorch_model.bin)，替换`bert-base-chinese`目录下的文件：

      ```shell
      mv pytorch_model.bin bert-base-chinese
      ```

   2. 导出onnx文件

      ```shell
      # 输入参数：${model_dir} ${output_path} ${seq_length} 
      python3 pth2onnx.py ./bert-base-chinese ./bert_base_chinese.onnx 384
      ```
      
      - 输入参数说明：第一个参数为源码仓路径（包含配置文件等），第二个参数为输出onnx文件路径，第三个参数为sequence长度。

   3. 优化onnx文件

      ```shell
      # 修改优化模型：${bs}:[1, 4, 8, 16, 32, 64],${seq_len}:384
      python3 -m onnxsim ./bert_base_chinese.onnx ./bert_base_chinese_bs${bs}.onnx --input-shape "input_ids:${bs},${seq_len}" "attention_mask:${bs},${seq_len}" "token_type_ids:${bs},${seq_len}"
      python3 fix_onnx.py bert_base_chinese_bs${bs}.onnx bert_base_chinese_bs${bs}_fix.onnx
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
         # bs:[1, 4, 8, 16, 32, 64]
         atc --model=./bert_base_chinese_bs${bs}_fix.onnx --framework=5 --output=./bert_base_chinese_bs${bs} --input_format=ND --log=debug --soc_version=${chip_name} --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance
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

   1. 安装ais_bench推理工具

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```
        # 以bs1模型推理为例
        mkdir -p ./output_data/bs1
        python3 -m ais_bench --model ./bert_base_chinese_bs1.om --input ./input_data/input_ids,./input_data/attention_mask,./input_data/token_type_ids --output ./output_data/ --output_dirname bs1 --batchsize 1 --device 1
        ```
        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入文件。
             -   --output：输出目录。
             -   --output_dirname：输出文件名。
             -   --device：NPU设备编号。
             -   --outfmt: 输出数据格式。
             -   --batchsize：推理模型对应的batchsize。


        推理后的输出默认在当前目录outputs/bs1下。

   3.  精度验证。

      调用postprocess.py脚本与数据集标签比对，获得Accuracy数据。

      ```
      # 以bs1模型推理为例
      # 输入参数：${result_dir} ${gt_dir} ${seq_length}
      python3 postprocess.py ./output_data/bs1 ./input_data/labels 384
      ```
      
      - 参数说明：第一个参数为推理结果路径，第二个参数为gt labe所在路径，第三个参数为sequence长度。

## 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

精度：

|       模型        |  Pth精度   | NPU离线推理精度 |
| :---------------: | :--------: | :-------------: |
| Bert-Base-Chinese | Acc:77.96% |   Acc: 77.94%   |

性能：

|       模型        | BatchSize | NPU性能 | 基准性能  |  基准性能2  |
| :---------------: | :-------: | :-----: | :-------: |:-------: |
| Bert-Base-Chinese |     1     | 175 fps | 41.16 fps |242 fps |
| Bert-Base-Chinese |     4     | 251 fps | 43.07 fps |307 fps |
| Bert-Base-Chinese |     8     | 254 fps | 42.82 fps |316 fps |
| Bert-Base-Chinese |    16     | 242 fps | 43.07 fps |333 fps |
| Bert-Base-Chinese |    32     | 246 fps | 42.32 fps |331 fps |
| Bert-Base-Chinese |    64     | 251 fps | 44.31 fps |336 fps |

## 其他下游任务<a name="ZH-CN_TOPIC_0000001126121892"></a>

+ [序列标注(Sequence Labeling)](downstream_tasks/sequence_labeling/README.md)
# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
