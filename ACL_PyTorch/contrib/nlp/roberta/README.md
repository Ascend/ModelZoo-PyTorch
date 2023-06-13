# RoBERTa模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>


`RoBERTa` 属于BERT的强化版本，也是BERT模型更为精细的调优版本。RoBERTa 模型是BERT 的改进版(从其名字来看，A Robustly Optimized BERT，即简单粗暴称为强力优化的BERT方法)。主要在在模型规模、算力和数据上，进行了一些改进。

- 参考实现：

  ```
  url=https://github.com/pytorch/fairseq.git
  mode_name=RoBERTa
  hash=c1624b27
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  说明：原仓默认的seq_length为70

  | 输入数据   | 数据类型 | 大小                      | 数据排布格式 |
  | --------   | -------- | ------------------------- | ------------ |
  | src_tokens | INT64    | batchsize x seq_len       | ND           |

- 输出数据

  | 输出数据 | 数据类型 | 大小                  | 数据排布格式 |
  | -------- | -------- | --------              | ------------ |
  | output   | FLOAT32  | batchsize x num_class | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动:

  **表 1**  版本配套表

  | 配套                                                            | 版本    | 环境准备指导                                                                                          |
  | ------------------------------------------------------------    | ------- | ------------------------------------------------------------                                          |
  | 固件与驱动                                                      | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 6.0.RC1 | -                                                                                                     |
  | Python                                                          | 3.7.5   | -                                                                                                     |
  | PyTorch                                                         | 1.5.0+  | -                                                                                                     |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git        # 克隆仓库的代码
   git checkout master         # 切换到对应分支
   cd ACL_PyTorch/contrib/nlp/roberta              # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   git clone https://gitee.com/ascend/msadvisor && cd msadvisor && git checkout master
   cd auto-optimizer && python3 -m pip install .
   cd ../..
   ```

   安装模型依赖:

   ```
   git clone https://github.com/pytorch/fairseq.git fairseq_workspace
   cd fairseq_workspace
   git checkout c1624b27
   git apply ../roberta-infer.patch
   pip3 install --editable ./
   ```

## 准备数据集<a name="section183221994411"></a>


   本模型使用 [SST-2官方数据集](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip)，解压到 `./data` 目录，如 `./data/SST-2` ,目录结构如下：

    ```
    ├── data
    |   ├── SST-2
    |   |    ├── test.tsv
    │   |    ├── dev.tsv
    │   |    ├── train.tsv
    │   |    ├── original/
    ```

   执行预处理脚本:

   ```
   # 使用代码仓自带脚本下载&&完成部分前置处理工作
   bash fairseq_workspace/examples/roberta/preprocess_GLUE_tasks.sh data/ SST-2
   若提示下载失败，则修改preprocess_GLUE_tasks.sh中wget部分代码
   将：
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
   修改为：
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json' --no-check-certificate
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe' --no-check-certificate
    wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt' --no-check-certificate
    

   # 生成om推理所需预处理数据
   python3 RoBERTa_preprocess.py --data_path ./data/SST-2-bin --pad_length 70
   ```

   - 参数说明：

     - --data_path: 数据集所在路径，输出数据保存在其下层目录下

     - --pad_length: 模型输入seq长度

  生成预处理数据在 `./data/SST-2-bin/roberta_base_bin_70`

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   获取权重文件：[RoBERTa模型pth权重文件](https://pan.baidu.com/s/1GZnnpz8fek2w7ARsZ0ujnA)，密码: x248。

   解压后将checkpoint.pt文件放至 `./checkpoints` 目录下(如没有则新建该目录)。

   1. 导出onnx文件。

      1. 使用以下脚本导出onnx文件:

         ```
         # 以bs1为例
         python3 RoBERTa_pth2onnx.py --checkpoint_path checkpoints/ --checkpoint_file checkpoint.pt --data_name_or_path ./data/SST-2-bin --onnx_path outputs --batch_size 1 --pad_length 70
         ```
         - 参数说明：

           - --checkpoint_path：权重文件所在目录

           - --checkpoint_file：权重文件名
           
           - --data_name_or_path: 数据集路径
           
           - --onnx_path: 输出onnx文件所在目录
           
           - --batch_size: 模型batchsize
           
           - --pad_length: 模型输入seq长度

         获得roberta_base_seq70_bs1.onnx文件。

      2. 优化ONNX文件。

         ```
         # 以bs1为例
         python3 -m onnxsim outputs/roberta_base_seq70_bs1.onnx outputs/roberta_base_seq70_bs1_sim.onnx
         # 输入参数: {原始模型} {修改后的模型路径} {batch_size} {seq_length}
         python3 opt_onnx.py outputs/roberta_base_seq70_bs1_sim.onnx outputs/roberta_base_seq70_bs1_opt.onnx 1 70
         ```

         获得roberta_base_seq70_bs1_opt.onnx文件。

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

         ```
         # bs1为例
         atc --framework=5 --model=./outputs/roberta_base_seq70_bs1_opt.onnx --output=./outputs/roberta_base_seq70_bs1 --input_format=ND --input_shape="src_tokens:1,70" --log=debug --soc_version=${chip_name} --op_precision_mode=precision.ini
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --op_precision_mode: 指定部分算子采用特定精度模式。

           运行成功后生成模型文件roberta_base_seq70_bs1.om。

2. 开始推理验证。

   1. 使用ais-bench工具进行推理。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        # 以bs1为例
        mkdir -p results/bs1
        python3 -m ais_bench --model outputs/roberta_base_seq70_bs1.om --input ./data/SST-2-bin/roberta_base_bin_70 --output results/ --output_dirname seq70_bs1 --device 1 --batchsize 1
        ```

        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入文件。
             -   --output：输出目录。
             -   --device：NPU设备编号。
             -   --batchsize: 模型对应batchsize。


        推理后的输出默认在当前目录results/seq70_bs1下。


   3. 精度验证。

      调用脚本与GT label，可以获得精度数据:

      ```
      # 以bs1为例
      python3 RoBERTa_postprocess.py --res_path=./results/seq70_bs1/ --data_path=./data/SST-2-bin
      ```

      - 参数说明：

        - --res_path：为生成推理结果所在路径

        - --data_path：为标签数据所在目录

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

   调用ACL接口推理计算，性能参考下列数据。

   默认seq_length为70的精度/性能如下：

   基准精度：ACC: 94.8%

   | 芯片型号 | Batch Size | 数据集 | 精度       | 性能      |
   |----------|------------|--------|------------|-----------|
   | 310P3    | 1          | SST-2  | Acc: 94.0% | 205 fps   |
   | 310P3    | 4          | SST-2  | -          | 817 fps   |
   | 310P3    | 8          | SST-2  | -          | 1244 fps  |
   | 310P3    | 16         | SST-2  | -          | 1463 fps  |
   | 310P3    | 32         | SST-2  | -          | 1473 fps  |
   | 310P3    | 64         | SST-2  | -          | 1206 fps  |
   | 310      | 1          | SST-2  | Acc: 94.4% | 12.01 fps |
   | 310      | 16         | SST-2  | -          | 98.49 fps |

   其他seq_length下部分精度性能如下（仅展示bs1/最优bs）:

   | seq_length | Batch Size | 数据集 | 基准精度   | 310P精度   | 310P性能 |
   |------------|------------|--------|------------|------------|----------|
   | 16         | 1          | SST-2  | Acc: 86.7% | Acc: 86.6% | 602fps   |
   | 16         | 64         | SST-2  | -          | -          | 8649fps  |
   | 32         | 1          | SST-2  | Acc: 93.8% | Acc: 93.2% | 508fps   |
   | 32         | 64         | SST-2  | -          | -          | 4718fps  |
   | 64         | 1          | SST-2  | Acc: 94.7% | Acc: 94.1% | 405fps   |
   | 64         | 32         | SST-2  | -          | -          | 2413fps  |
   | 128        | 1          | SST-2  | Acc: 94.7% | Acc: 94.1% | 418fps   |
   | 128        | 32         | SST-2  | -          | -          | 1100fps  |
