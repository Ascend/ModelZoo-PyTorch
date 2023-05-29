# Bert_Base_Uncased模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

BERT，即Bidirectional Encoder Representations from Transformers，是一种基于Transformer的自然语言处理预训练模型，由Google于2018年发布。当时它在许多自然语言任务中表现出了卓越的性能，之后也成为了几乎所有NLP研究中的性能基线。本文使用的是BERT_base模型。


- 参考实现：

  ```
  url        = https://github.com/NVIDIA/DeepLearningExamples.git
  commit_id  = dd6b8ca2bb80e17b015c0f61e71c2a84733a5b32
  code_path  = DeepLearningExamples/PyTorch/LanguageModeling/BERT/
  model_name = BERTBASE
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  |  输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | :-------: | :----: | :-------------: | :-------: |
  |input_ids  | INT64  | batchsize × 512 | ND          |
  |segment_ids| INT64  | batchsize × 512 | ND          |
  |input_mask | INT64  | batchsize × 512 | ND          |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | :-------: | :----: | :-------------: | :-------: |
  | output   | INT64  | batchsize × 512 | ND     |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.4  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 |        |                                                             |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>


1. 获取本仓代码

   ```bash
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd ./ModelZoo-PyTorch/ACL_PyTorch/built-in/nlp/Bert_Base_Uncased_for_Pytorch/
   ```

   文件说明
   ```
   Bert_Base_Uncased_for_Pytorch
      ├── bert_config.json                 //bert_base模型网络配置参数
      ├── bert_base_get_info.py            //生成推理输入的数据集二进制info文件
      ├── bert_preprocess_data.py          //数据集预处理脚本，生成二进制文件
      ├── ReadMe.md                        //此文档
      ├── bert_base_uncased_atc.sh         //onnx模型转换om模型脚本
      ├── bert_base_pth2onnx.py            //用于转换pth模型文件到onnx模型文件
      ├── bert_postprocess_data.py         //bert_base数据后处理脚本，用于将推理结果处理映射成文本
      └── evaluate_data.py                 //验证推理结果脚本，比对ais_bench输出的分类结果，给出accuracy
   ```

2. 安装依赖

   ```bash
   pip3 install -r requirements.txt
   ```

3. 安装改图工具依赖
   ```bash
   git clone https://gitee.com/Ronnie_zheng/MagicONNX.git MagicONNX
   cd MagicONNX && git checkout 99a713801fe70db702e3903744d2a6372a975fea
   pip3 install . && cd ..
   ```

4. 获取BERT源码

   ```bash
   git clone https://github.com/NVIDIA/DeepLearningExamples.git        
   cd ./DeepLearningExamples             
   git reset --hard dd6b8ca2bb80e17b015c0f61e71c2a84733a5b32
   cd ..               
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型支持使用squad QA的验证集。
   
   以squad v1.1为例，执行以下指令获取[squad v1.1](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)数据集。

   ```bash
   wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O ./dev-v1.1.json --no-check-certificate
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   运行```bert_preprocess_data.py```脚本，执行数据预处理。
      ```
      python3 bert_preprocess_data.py --max_seq_length=512 --do_lower_case --vocab_file=./DeepLearningExamples/PyTorch/LanguageModeling/BERT/vocab/vocab --predict_file=./dev-v1.1.json --save_dir ./bert_bin
      ```

   参数说明：

      - --max_seq_length：句子最大长度。
      - --vocab_file：数据字典映射表文件。
      - --do_lower_case：是否进行大小写转化。
      - --predict_file：原始验证数据文本，将后处理数据位置映射到原始文件。
      - --save_dir：转换结果的输出文件夹路径。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      在PyTorch开源框架中获取[bert_base_qa.pt](https://catalog.ngc.nvidia.com/orgs/nvidia/models/bert_pyt_ckpt_base_qa_squad11_amp/files)文件。

      ```bash
      wget 'https://api.ngc.nvidia.com/v2/models/nvidia/bert_pyt_ckpt_base_qa_squad11_amp/versions/19.09.0/files/bert_base_qa.pt' -O ./bert_base_qa.pt --no-check-certificate
      ```

   2. 导出onnx文件（以batch_size=8为例）。

      1. 使用```bert_base_pth2onnx.py```导出onnx文件。


         ```bash
         python3 bert_base_pth2onnx.py --init_checkpoint=bert_base_qa.pt --save_dir ./ --config_file=bert_config.json --batch_size=8
         ```

         参数说明：

         - --init_checkpoint：输入权重文件。
         - --save_dir：所导出onnx模型会保存在该目录下。
         - --config_file：网络参数配置文件。
         - --batch_size：批次大小。


         运行成功后，在当前目录下会生成```bert_base_batch_8.onnx```模型文件（```_8```表示batch_size为8）

      2. 简化、修改onnx文件

         ```bash
         # 简化onnx文件
         python3 -m onnxsim bert_base_batch_8.onnx bert_base_batch_8_sim.onnx
    
         # 修改onnx文件
         python3 fix_onnx.py bert_base_batch_8_sim.onnx bert_base_batch_8_fix.onnx
         ```

         获得```bert_base_batch_8_fix.onnx```文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```bash
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

      3. 执行ATC命令（注意替换指令中的```${batch_size}```和```${chip_name}```）。

         ```
          atc --input_format=ND --framework=5 --model=bert_base_batch_${batch_size}_fix.onnx --input_shape="input_ids:${batch_size},512;token_type_ids:${batch_size},512;attention_mask:${batch_size},512" --output=bert_base_batch_${batch_size}_auto --log=error --soc_version=${chip_name}  --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --insert\_op\_conf=aipp\_resnet34.config:  AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用，详见下文数据预处理。

           运行成功后得到用于二进制输入推理的模型文件```bert_base_batch_8_auto.om```。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理（以batch_size=8为例）。

        ```bash
      #完整推理
      python3 -m ais_bench --model ./bert_base_batch_8_auto.om --input ./bert_bin/input_ids,./bert_bin/segment_ids,./bert_bin/input_mask --batchsize 8 --output ./bert_bin --output_dirname outputs
      ```

        -   参数说明：

             -   --model：为ONNX模型文件。
             -   --batchsize：批次大小。
             -   --input：模型的输入文件夹路径。
             -   --output：推理结果输出路径。
             -   --output_dirname：输出子文件夹，与参数output搭配使用，结果将保存到 output/output_dirname文件夹中。

        推理结果输出在目录```./bert_bin/outputs/```下。


   3. 精度验证。

      1. 推理结果后处理

         将结果转化为json文本数据，执行命令如下。

         ```bash
         python3 bert_postprocess_data.py --max_seq_length=512 --vocab_file=./DeepLearningExamples/PyTorch/LanguageModeling/BERT/vocab/vocab --do_lower_case --predict_file=./dev-v1.1.json --npu_result=./bert_bin/outputs/
         ```

         参数说明：

         - --max_seq_length：句子最大长度。
         - --vocab_file：数据字典映射表文件。
         - --do_lower_case：是否进行大小写转化。
         - --predict_file：原始验证数据文本，将后处理数据位置映射到原始文件。
         - --npu_result：推理结果目录。

      2. 精度验证

         执行```evaluate_data.py```脚本将原始数据dev-v1.1.json与推理结果数据文本predictions.json比对，可以获得Accuracy数据，执行命令如下。

         ```
         python3 evaluate_data.py dev-v1.1.json predictions.json
         ```

         参数说明：
         - dev-v1.1.json：为标签数据
         - predictions.json：为推理结果文件  


   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```bash
        #执行纯推理命令，验证模型性能：
      python3 -m ais_bench --model ./bert_base_batch_8_auto.om --batchsize 8 --loop 100
        ```

       参数说明：
         -   --model：为ONNX模型文件。
         -   --batchsize：批次大小。
         -   --loop：推理次数。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 | 基准性能 |
| :-------: | :--------------: | :--------: | :--------: | :-------------: | :---------:|
| 310P3   |    1            |  SQuAD v1.1 |     88.78%       |  177.81 fps  | 158.69 fps |
| 310P3   |    4            |  SQuAD v1.1 |   88.78%      |   221.81 fps    | 176.86 fps |
| 310P3   |    8            |  SQuAD v1.1 |     -       |   218.12 fps    | 181.41 fps |
| 310P3   |    16            |  SQuAD v1.1 |    -        |  218.71 fps   | 199.48 fps |
| 310P3   |    32            |  SQuAD v1.1 |    -        |  218.48 fps |206.35 fps |
| 310P3   |    64            |  SQuAD v1.1 |    -        |  211.73 fps  |197.44 fps |
