#   SpanBERT 模型-推理指导


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

SpanBERT在BERT的基础上，采用Geometric Spans的遮盖方案并加入Span Boundary Objective (SBO) 训练目标，通过使用分词边界的表示以及被遮盖词元的位置向量来预测被遮盖的分词的内容，增强了 BERT 的性能，特别是在一些与 Span 相关的任务，如抽取式问答。


- 参考实现：

  ```
  url=https://github.com/facebookresearch/SpanBERT.git
  commit_id=96f2dfbede280df3a5d146425a9c8eca7b425d41
  code_path=/ACL_PyTorch/contrib/nlp/SpanBERT
  model_name=SpanBERT
  ```
  





## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小            | 数据排布格式 |
  | -------- | -------- | --------------- | ------------ |
  | input1   | FLOAT32  | batchsize x 512 | ND           |
  | input2   | FLOAT32  | batchsize x 512 | ND           |
  | input3   | FLOAT32  | batchsize x 512 | ND           |


- 输出数据

  | 输出数据 | 数据类型 | 大小            | 数据排布格式 |
  | -------- | -------- | --------------- | ------------ |
  | output   | FLOAT32  | batchsize x 512 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

- 该模型需要以下依赖  

  **表 2**  依赖列表

  | 依赖名称    | 版本   |
  | ----------- | ------ |
  | torch       | 1.8.1  |
  | boto3       | 1.24.3 |
  | requests    | 2.27.1 |
  | tqdm        | 4.64.0 |
  | sympy       | 1.10.1 |
  | decorator   | 5.1.1  |
  | onnxruntime | 1.11.1 |
  | numpy       | 1.21.6 |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

    ```
   git clone https://github.com/facebookresearch/SpanBERT.git
   cd SpanBERT
   git checkout 0670d8b6a38f6714b85ea7a033f16bd8cc162676
   cd ..
   ```
   
2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持[SQuAD 1.1](https://github.com/rajpurkar/SQuAD-explorer/tree/master/datase)数据集， 上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到dev-v1.1.json验证集。目录结构如下：

   ```
   ├── SQuAD1.1
     ├── dev-v1.1.json
     ├── train-v1.1.json
   ```

2. 数据预处理。

   将原始数据集转换为模型输入的数据。

   执行脚本 spanBert_preprocess.py 。

   ```
   python3 spanBert_preprocess.py \
       --dev_file ${datasets_path}/dev-v1.1.json 
   ```
   
   + 参数说明：
     + --dev_file：原数据集的路径。
   
   执行该命令后将会得到如下结构的文件夹。
   
   ```
   ├── input_ids 
   │    ├──0.bin
   │    ├──......     	 
   ├── input_mask 
   │    ├──0.bin
   │    ├──...... 
   ├── segment_ids 
   │    ├──0.bin
   │    ├──......
   ```
   
   


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
        执行下方代码获取权重文件

        ```
        data_dir=$1
        model=$2
        echo Downloading $model
        wget -P $data_dir http://dl.fbaipublicfiles.com/fairseq/models/spanbert_$model.tar.gz
        mkdir $data_dir/$model
        tar xvzf $data_dir/spanbert_$model.tar.gz -C $data_dir/$model
        rm $data_dir/spanbert_$model.tar.gz
        ```
        
        + 参数说明：
          + data_dir：权重下载后存放的文件夹,默认使用model_dir。
          + model：为任务名，默认使用squad1。

   2. 导出onnx文件。

      1. 使用  spanBert_pth2onnx.py 导出onnx文件。
   
         运行 spanBert_pth2onnx.py 脚本,
   
         ```
          python3 spanBert_pth2onnx.py  \
         --config_file ./model_dir/squad1/config.json  \
         --checkpoint ./model_dir/squad1/pytorch_model.bin
         ```
   
         + 参数说明：
           + --config_file： 模型config文件的路径 。
           + --checkpoint： 表示模型bin文件的路径 。
         
         执行后在当前路径下生成 spanBert_dynamicbs.onnx 模型文件。 
   
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
         atc --framework=5 --model=./spanBert_dynamicbs.onnx --output=./spanBert_bs1 --input_format=ND --input_shape="input_ids:1,512;token_type_ids:1,512;attention_mask:1,512" --log=error --soc_version=Ascend${chip_name}
         ```
         
         - 参数说明：
         
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           
         
         运行成功后生成 spanBert_bs1.om 模型文件。
   
2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```
        python3 -m ais_bench --model ./spanBert_bs1.om --input "./input_ids,./segment_ids,./input_mask" --output ./result/bs1 --outfmt BIN --batchsize 1
        ```
        
        - 参数说明：
          - --model：om模型。
          - --input：预处理数据集路径。
          - --output：推理结果所在路径。
          - --outfmt：推理结果文件格式。
          - --batchsize：不同的batchsize。
        
        推理后的输出默认在当前目录result下。
        
   
   3. 精度验证。
   
       调用 spanBert_postprocess.py 进行精度计算。
       
       ```
         python3 spanBert_postprocess.py \
          	--do_eval \
          	--model spanbert-base-cased \
          	--dev_file /opt/npu/squad1/dev-v1.1.json \
          	--max_seq_length 512 \
          	--doc_stride 128 \
          	--eval_metric f1 \
          	--fp16 \
          	--bin_dir ./result/bs1/ \
       ```
       
       - 参数说明：
       
         + --do_eval：提供此参数程序会将输入数据进行验证。
         + --model：使用模型，默认：spanbert-base-cased 。
       
         - --dev_file  ： squad1验证集路径。
         - --max_seq_length：最长序列长度，默认：512 。
         - --doc_stride：文档步长，默认：128。
         - --eval_metric：评估度量，默认：f1。
         - --fp16：是否使用混合精度。
         - --bin_dir ：  推理输出所在的文件夹 。
   
   4. 性能验证。
   
      调用ACL接口推理，将{batchsize}改为实际的batchsize。参考命令如下：
      
      ```
       python3 -m ais_bench --model /spanBert_bs16.om --output ./lcmout/ --output_dirname bs1 --outfmt BIN --batchsize 16 --loop 5
      ```
      
      + 参数说明：
        + --model：om文件路径。
        + --output：推理结果所在路径。
        + --outfmt：推理结果文件格式。
        + --output_dirname： 推理结果输出子文件夹。可选参数。与参数output搭配使用，单独使用无效。设置该值时输出结果将保存到 output/output_dirname文件夹中 。
        + --batchsize：不同的batchsize。
        + --loop：推理的次数。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| Precision |           |
| --------- | --------- |
| 标杆精度  | f1:92.4%  |
| 310P3精度 | f1:93.95% |

| 芯片型号 | Batch Size   | 数据集 | 性能 |
| --------- | ---------------- | ---------- | --------------- |
| 310P3 | 1 | SQuAD 1.1 | 43.833   |
| 310P3 | 4 | SQuAD 1.1 | 25.411 |
| 310P3 | 8 | SQuAD 1.1 | 31.726 |
| 310P3 | 16 | SQuAD 1.1 | 31.904 |
| 310P3 | 32 | SQuAD 1.1 | 32.038 |
| 310P3 | 64 | SQuAD 1.1 | 32.343 |

