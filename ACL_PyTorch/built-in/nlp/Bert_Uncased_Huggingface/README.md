#   BERT_Uncased_Huggingface模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)


  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

------

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

BERT（Bidirectional Encoder Representations from Transformers）是一种预训练语言表示（Language Representation）的方法，通过在一个大型文本语料库上训练一个通用的语言理解模型，然后将该模型应用于不同的下游任务中。本文档以语音问答（Question Answering）任务为例，支持如下配置BERT模型的推理：

- 支持 BERT-Base 和 BERT-Large 两种 model size（[具体差异](https://github.com/google-research/bert#pre-trained-models)）。
- 支持不同 sequence length 和 batch size 的静态模型推理。

参考论文：

- [Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

参考实现：

- 模型结构

  ```
  https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/models/bert/modeling_bert.py#L1796
  ```

- 模型配置和权重

  ```
  # BERT-Base
  https://huggingface.co/csarron/bert-base-uncased-squad-v1
  # BERT-Large
  https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据       | 数据类型 | 大小                | 数据排布格式 |
  | -------------- | -------- | ------------------- | ------------ |
  | input_ids      | INT64    | batchsize x seq_len | ND           |
  | attention_mask | INT64    | batchsize x seq_len | ND           |
  | token_type_ids | INT64    | batchsize x seq_len | ND           |

- 输出数据

  | 输出数据     | 数据类型 | 大小                | 数据排布格式 |
  | ------------ | -------- | ------------------- | ------------ |
  | start_logits | FLOAT32  | batchsize x seq_len | ND           |
  | end_logits   | FLOAT32  | batchsize x seq_len | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1** 版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 23.0.RC2  | [Pytorch框架推理环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fpies) |
  | CANN                                                         | 6.3.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取本仓源码。

   ```shell
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git        # 克隆仓库的代码
   git checkout master                                            # 切换到对应分支
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/nlp/Bert_Uncased_Huggingface           # 切换到模型的代码仓目录
   ```

2. 获取模型配置文件，和第1步源码置于同级目录下。

   ```shell
   # Bert-Base
   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csarron/bert-base-uncased-squad-v1
   mv ./bert-base-uncased-squad-v1 ./bert_base
   
   # Bert-Large
   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad
   mv ./bert-large-uncased-whole-word-masking-finetuned-squad ./bert_large
   ```

   所代码克隆失败，可尝试把 `https` 替换成 `http` 重试。

3. 安装必要依赖。

   ```shell
   pip3 install -r requirements.txt
   ```

4. 安装改图依赖 [auto-optimizer](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer)。

   ```shell
   git clone https://gitee.com/ascend/msadvisor.git
   cd msadvisor/auto-optimizer
   pip3 install -r requirements.txt
   python3 setup.py install&&cd ../..
   ```

5. 前后处理的步骤涉及在线下载数据集，若遇SSL校验错误的情况，需配置环境变量如下：

   ```shell
   export CURL_CA_BUNDLE=
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   该模型使用 [SQuAD v1.1](https://huggingface.co/datasets/squad) 数据集，下载过程已包含在 bert_process.py 处理脚本中。若因网络原因无法下载，需预先[下载](https://pan.baidu.com/s/1aph8lNX9WOT5e7Yf8doWhg?pwd=data)数据集相关文件，解压后和源码置于同级目录下。

   ```
   <project>
   |-- squad
   |   |-- data
   |   |   |-- dataset_dict.json
   |   |   |-- train
   |   |   |   |-- data-00000-of-00001.arrow
   |   |   |   |-- dataset_info.json
   |   |   |   `-- state.json
   |   |   `-- validation
   |   |       |-- data-00000-of-00001.arrow
   |   |       |-- dataset_info.json
   |   |       `-- state.json
   |   `-- metric
   |       |-- evaluate.py
   |       `-- squad.py
   |-- README.md
   |-- ......
   ```

2. 执行 bert_process.py 脚本，生成数据集预处理后的npy文件

   ```shell
   # model_size = [base, large]
   # seq = [64, 128, 256, 320, 384, 512]
   # stride = [16, 64, 128, 128, 128, 128]
   
   python3 bert_process.py \
   --model_path bert_${model_size} \
   --process_mode preprocess \
   --save_dir prep_data \
   --pad_to_max_length \
   --max_seq_length ${seq} \
   --doc_stride ${stride} \
   # --offline_download
   ```
   
   参数说明：
   
   - --model_path：模型配置和权重文件所在文件夹路径。
   - --process_mode：此处为**预处理模式**。
   - --save_dir：预处理后数据的保存路径。
   - --pad_to_max_length：静态模型需要添加此参数。
   - --max_seq_length：序列长度 `${seq}`。
   - --doc_stride：`${stride}`值与`${seq}`值一一对应。
   - --offline_download：若采用离线下载模式下载数据集，需要添加此参数。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      - 点击下载 Bert-Base 权重文件 [pytorch_model.bin](https://huggingface.co/csarron/bert-base-uncased-squad-v1/blob/main/pytorch_model.bin)， 替换 `bert_base` 目录下文件。
      - 点击下载 Bert-Large 权重文件 [pytorch_model.bin](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/pytorch_model.bin)， 替换 `bert_large` 目录下文件。

   2. 导出onnx文件。

      执行如下命令，采用 [transformers.onnx](https://huggingface.co/docs/transformers/serialization#export-to-onnx) 模块导出动态onnx文件。

      ```shell
      # BERT-Base
      python3 -m transformers.onnx --model=./bert_base/ --feature=question-answering onnx/
      mv onnx/model.onnx onnx/bert_base_dynamic.onnx
      
      # BERT-Large
      python3 -m transformers.onnx --model=./bert_large/ --feature=question-answering onnx/
      mv onnx/model.onnx onnx/bert_large_dynamic.onnx
      ```

      参数说明：

      - --model：模型权重和配置文件路径。
      - --feature：指定导出模型的对应下游任务特征。
      - --位置参数：指定文件夹路径，在该指定路径下生成model.onnx文件。

   3. 简化并转换为静态onnx文件。

      ```shell
      # model_size = [base, large]
      # seq = [64, 128, 256, 320, 384, 512]
      # bs = [1, 4, 8, 16, 32, 64]
      
      python3 -m onnxsim onnx/bert_${model_size}_dynamic.onnx onnx/bert_${model_size}_seq${seq}_bs${bs}.onnx \
      --input-shape "input_ids:${bs},${seq}" "attention_mask:${bs},${seq}" "token_type_ids:${bs},${seq}"
      ```

      参数说明：

      - --位置参数1：简化前onnx文件。
      - --位置参数2：简化后onnx文件。
      - --input-shape：指定输入维度信息。

   4. onnx模型优化。
   
      静态onnx模型优化：

      ```shell
      # model_size = [base, large]
      # seq = [64, 128, 256, 320, 384, 512]
      # bs = [1, 4, 8, 16, 32, 64]
      
      python3 fix_onnx.py \
      --input_file onnx/bert_${model_size}_seq${seq}_bs${bs}.onnx \
      --output_file onnx/bert_${model_size}_seq${seq}_bs${bs}_fix.onnx \
      --model_size ${model_size}
      ```

      参数说明：

      - --input_file：修改前onnx文件。
      - --output_file：修改后onnx文件。
      - --model_size：指定模型大小为 base 或 large。

      动态onnx模型优化（Transformer加速库优化）：
      
      ```shell
      # model_size = [base, large]
      python3 fix_onnx2unpad.py \
      --input_file onnx/bert_${model_size}_dynamic.onnx \
      --output_file onnx/bert_${model_size}_unpad.onnx
      ```
      
      参数说明：
      
      - --input_file: 原始动态onnx文件。
      - --output_file: 修改后onnx文件。
      
   5. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
          # 使能transformer加速库：动态Unpad方案必需
          source ${ASCENDIE_HOME}/set_env.sh
         ```

      2. 执行命令查看芯片名称（${chip_name}）。

         ```
         npu-smi info
         # 该设备芯片名为Ascend310P3 （自行替换）
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

   6. 执行ATC命令。
   
      静态模型转化：

      ```shell
      # model_size = [base, large]
      # seq = [64, 128, 256, 320, 384, 512]
      # bs = [1, 4, 8, 16, 32, 64]
      
      atc --model onnx/bert_${model_size}_seq${seq}_bs${bs}_fix.onnx \
      --output om/bert_${model_size}_seq${seq}_bs${bs} \
      --framework 5 \
      --log=error \
      --soc_version Ascend${chip_name} \
      --optypelist_for_implmode="Gelu" \
      --op_select_implmode=high_performance
      ```

      运行成功后生成 `bert_${model_size}_seq${seq}_bs${bs}.om` 模型文件。

      参数说明：
      
      - --model：为ONNX模型文件。
      - --output：输出的OM模型。
      - --framework：5代表ONNX模型。
      - --log：日志级别。
      - --soc_version：处理器型号。
      - --optypelist_for_implmode：指定算子类型。
      - --op_select_implmode：与optypelist_for_implmode配合使用，指定算子的实现模式。
      
      动态模型转化：

      ```shell
      # model_size = [base, large]
      atc --model onnx/bert_${model_size}_unpad.onnx \
      --output om/bert_${model_size}_unpad \
      --framework 5 \
      --log=error \
      --soc_version Ascend${chip_name} \
      --optypelist_for_implmode="Gelu" \
      --op_select_implmode=high_performance \
      --input_shape="input_ids:-1,384;attention_mask:-1,384;token_type_ids:-1,384"
      ```

      运行成功后生成 `bert_${model_size}_unpad_${os}_${arch}.om` 模型文件。

      参数说明：
      
      - --model：为ONNX模型文件。
      - --output：输出的OM模型。
      - --framework：5代表ONNX模型。
      - --log：日志级别。
      - --soc_version：处理器型号。
      - --optypelist_for_implmode：指定算子类型。
      - --op_select_implmode：与optypelist_for_implmode配合使用，指定算子的实现模式。
      - --input_shape: 模型的输入shape信息。
      
2. 开始推理验证。

   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)]

   2. 执行推理。

      静态模型推理：
  
      ```shell
      # model_size = [base, large]
      # seq = [64, 128, 256, 320, 384, 512]
      # bs = [1, 4, 8, 16, 32, 64]
      
      python3 -m ais_bench --model=om/bert_${model_size}_seq${seq}_bs${bs}.om  --batchsize=${bs} \
      --input ${prep_data}/input_ids,${prep_data}/attention_mask,${prep_data}/token_type_ids \
      --output result --output_dirname result_bs${bs} --outfmt NPY
      ```
      
      参数说明：
      
      - --model：om模型路径。
      - --batchsize：批次大小。
      - --input：输入数据所在路径。
      - --output：推理结果输出路径。
      - --output_dirname：推理结果输出子文件夹。
      - --outfmt：推理结果输出格式
      
      动态模型推理：
      
        ```shell
      # model_size = [base, large]
      # bs = [4, 8, 16, 32]
      
      python3 -m ais_bench --model=om/bert_${model_size}_unpad_${os}_${arch}.om \
      --input ${prep_data}/input_ids,${prep_data}/attention_mask,${prep_data}/token_type_ids \
      --output result_unpad --output_dirname result_bs${bs} --outfmt NPY \
      --dymShape "input_ids:32,384;attention_mask:32,384;token_type_ids:32,384" --outputSize 1000000,1000000
      ```
      
      参数说明：
      
      - --model：om模型路径。
      - --input：输入数据所在路径。
      - --output：推理结果输出路径。
      - --output_dirname：推理结果输出子文件夹。
      - --outfmt：推理结果输出格式。
      - --dymShape: 动态模型输入shape。
      - --outputSize： 动态模型输出buffer大小。
   
3. 精度验证。

   调用 bert_process.py 脚本获得精度数据。

   ```shell
   # model_size = [base, large]
   # seq = [64, 128, 256, 320, 384, 512]
   # bs = [1, 4, 8, 16, 32, 64]
   # stride = [16, 64, 128, 128, 128, 128]
   
   python3 bert_process.py \
   --model_path bert_${model_size} \
   --process_mode postprocess \
   --save_dir prep_data \
   --pad_to_max_length \
   --max_seq_length ${seq} \
   --result_dir ./result/result_bs${bs}
   --doc_stride ${stride}
   # --offline_download
   ```
   
   参数说明：
   
   - --model_path：模型配置和权重文件所在文件夹路径。
   - --process_mode：此处为**后处理模式**。
   - --pad_to_max_length：静态模型需要添加此参数。
   - --max_seq_length：序列长度 `${seq}`。
   - --result_dir：推理结果所在路径。
   - --doc_stride：`${stride}`值与`${seq}`值一一对应，需要与前处理保持一致。
   - --offline_download：若采用离线下载模式下载数据集，需要添加此参数。
   
4. 可使用ais_bench推理工具的纯推理模式验证不同batch_size的静态模型的性能，参考命令如下：

   ```shell
   # model_size = [base, large]
   # seq = [64, 128, 256, 320, 384, 512]
   # bs = [1, 4, 8, 16, 32, 64]
   
   python3 -m ais_bench --model=om/bert_${model_size}_seq${seq}_bs${bs}.om --loop=50 --batchsize=${bs}
   ```
   
   参数说明：
   
   - --model：om模型路径。
   - --batchsize：批次大小。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

### BERT-Base 模型

- 精度数据参考（数据集推理场景）：采用 SQuAD v1.1 验证集

| 模型方案 | 芯片型号    | Sequence Length | Doc Stride | pth 精度（ F1 \| EM ）                                                     | NPU 精度（ F1 \|EM ） |
|----------|-------------|-----------------|------------|----------------------------------------------------------------------------|-----------------------|
| 静态     | Ascend310P3 | 64              | 16         |  83.84 \| 76.43                                                            | 83.90 \| 76.42        |
| 静态     | ...         | 128             | 64         |  86.77 \| 79.56                                                            | 86.96 \| 79.85        |
| 静态     | ...         | 256             | 128        |  87.99 \| 80.64                                                            | 87.96 \| 80.58        |
| 静态     | ...         | 320             | 128        |  80.86 \| 88.20                                                            | 88.15 \| 80.77        |
| 静态     | ...         | 384             | 128        | [88.2  \| 80.9](https://huggingface.co/csarron/bert-base-uncased-squad-v1) | 88.20 \| 80.84        |
| 静态     | ...         | 512             | 128        |  88.10 \| 80.77                                                            | 88.19 \| 80.80        |
| 动态     | ...         | 384             | 128        | [88.2  \| 80.9](https://huggingface.co/csarron/bert-base-uncased-squad-v1) | 87.98 \| 80.64        |
| 动态     | ...         | 512             | 128        |  88.10 \| 80.77                                                            | 87.95 \| 80.59        |

- 静态性能数据参考（纯推理场景）

| 芯片型号    | Sequence Length | Batch Size | NPU 性能（FPS） |
| ----------- | --------------- | ---------- | --------------- |
| Ascend310P3 | 64              | 1          | 421.30          |
| ...         | ...             | 4          | 1672.77         |
| ...         | ...             | 8          | 2507.14         |
| ...         | ...             | 16         | 2699.84         |
| ...         | ...             | 32         | 2883.59         |
| ...         | ...             | 64         | 2641.95         |
| ...         | 128             | 1          | 484.39          |
| ...         | ...             | 4          | 1166.12         |
| ...         | ...             | 8          | 1321.87         |
| ...         | ...             | 16         | 1344.72         |
| ...         | ...             | 32         | 1273.30         |
| ...         | ...             | 64         | 1242.68         |
| ...         | 256             | 1          | 336.10          |
| ...         | ...             | 4          | 547.37          |
| ...         | ...             | 8          | 577.30          |
| ...         | ...             | 16         | 569.14          |
| ...         | ...             | 32         | 555.13          |
| ...         | ...             | 64         | 542.94          |
| ...         | 320             | 1          | 271.03          |
| ...         | ...             | 4          | 385.70          |
| ...         | ...             | 8          | 372.05          |
| ...         | ...             | 16         | 357.51          |
| ...         | ...             | 32         | 374.45          |
|             |                 | 64         | 363.87          |
| ...         | 384             | 1          | 224.96          |
| ...         | ...             | 4          | 328.64          |
| ...         | ...             | 8          | 316.07          |
| ...         | ...             | 16         | 308.66          |
| ...         | ...             | 32         | 314.01          |
| ...         | ...             | 64         | 313.14          |
| ...         | 512             | 1          | 178.97          |
| ...         | ...             | 4          | 221.47          |
| ...         | ...             | 8          | 220.62          |
| ...         | ...             | 16         | 220.08          |
| ...         | ...             | 32         | 218.18          |
| ...         | ...             | 64         | 212.47          |

- 动态性能数据参考（数据集推理）

| 芯片型号    | 数据集     | Sequence Length | Batch Size | NPU 性能（FPS） |
| ----------- | ------     | ---------       | ---------- | --------------- |
| Ascend310P3 | SQuAD v1.1 | 384             | 4          | 374             |
| ...         | ...        | ...             | 8          | 421             |
| ...         | ...        | ...             | 16         | 440             |
| ...         | ...        | ...             | 32         | 447             |
| ...         | ...        | ...             | 64         | 451             |
| ...         | ...        | 512             | 4          | 307             |
| ...         | ...        | ...             | 8          | 372             |
| ...         | ...        | ...             | 16         | 416             |
| ...         | ...        | ...             | 32         | 428             |
| ...         | ...        | ...             | 64         | 436             |

### BERT-Large 模型

- 精度数据参考（数据集推理场景）：采用 SQuAD v1.1 验证集

| 模型方案 | 芯片型号    | Sequence Length | Doc Stride | pth 精度（ F1 \| EM ）                                                                          | NPU 精度（ F1 \|EM ） |
|----------|-------------|-----------------|------------|-------------------------------------------------------------------------------------------------|-----------------------|
| 静态     | Ascend310P3 | 64              | 16         |  88.74 \| 82.10                                                                                 | 88.51 \| 81.79        |
| 静态     | ...         | 128             | 64         |  92.08 \| 85.87                                                                                 | 91.92 \| 85.82        |
| 静态     | ...         | 256             | 128        |  93.06 \| 86.84                                                                                 | 93.01 \| 86.84        |
| 静态     | ...         | 320             | 128        |  93.15 \| 86.92                                                                                 | 93.00 \| 86.90        |
| 静态     | ...         | 384             | 128        | [93.15 \| 86.91](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad)  | 93.10 \| 86.94        |
| 静态     | ...         | 512             | 128        |  93.16 \| 86.93                                                                                 | 93.10 \| 86.94        |
| 动态     | ...         | 384             | 128        | [93.15 \| 86.91](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad)  | 92.99 \| 86.83        |
| 动态     | ...         | 512             | 128        |  93.16 \| 86.93                                                                                 | 93.10 \| 86.94        |

- 静态性能参考数据（纯推理场景）

| 芯片型号    | Sequence Length | Batch Size | NPU 性能（FPS） |
| ----------- | --------------- | ---------- | --------------- |
| Ascend310P3 | 64              | 1          | 204.1607        |
| ...         | ...             | 4          | 572.3639        |
| ...         | ...             | 8          | 694.0155        |
| ...         | ...             | 16         | 762.6932        |
| ...         | ...             | 32         | 768.5499        |
| ...         | ...             | 64         | 746.2670        |
| ...         | 128             | 1          | 188.9719        |
| ...         | ...             | 4          | 332.5935        |
| ...         | ...             | 8          | 367.1520        |
| ...         | ...             | 16         | 364.3270        |
| ...         | ...             | 32         | 353.7399        |
| ...         | ...             | 64         | 346.8615        |
| ...         | 256             | 1          | 125.5092        |
| ...         | ...             | 4          | 162.0578        |
| ...         | ...             | 8          | 163.4326        |
| ...         | ...             | 16         | 158.2651        |
| ...         | ...             | 32         | 155.5583        |
| ...         | ...             | 64         | 154.2266        |
| ...         | 320             | 1          | 96.4345         |
| ...         | ...             | 4          | 119.5064        |
| ...         | ...             | 8          | 125.2701        |
| ...         | ...             | 16         | 114.0485        |
| ...         | ...             | 32         | 101.7498        |
| ...         | ...             | 64         | 101.2074        |
| ...         | 384             | 1          | 78.6289         |
| ...         | ...             | 4          | 101.1932        |
| ...         | ...             | 8          | 98.6971         |
| ...         | ...             | 16         | 98.2296         |
| ...         | ...             | 32         | 98.1087         |
| ...         | ...             | 64         | 98.8876         |
| ...         | 512             | 1          | 62.2018         |
| ...         | ...             | 4          | 70.7663         |
| ...         | ...             | 8          | 71.3450         |
| ...         | ...             | 16         | 71.5686         |
| ...         | ...             | 32         | 70.5622         |
| ...         | ...             | 64         | 70.0256         |

- 动态性能数据参考（数据集推理）

| 芯片型号    | 数据集     | Sequence Length | Batch Size | NPU 性能（FPS） |
| ----------- | ------     | ---------       | ---------- | --------------- |
| Ascend310P3 | SQuAD v1.1 | 384             |          4 |             133 |
| ...         | ...        | ...             |          8 |             142 |
| ...         | ...        | ...             |         16 |             144 |
| ...         | ...        | ...             |         32 |             140 |
| ...         | ...        | ...             |         64 |             141 |
| Ascend310P3 | SQuAD v1.1 | 512             |          4 |             116 |
| ...         | ...        | ...             |          8 |             130 |
| ...         | ...        | ...             |         16 |             137 |
| ...         | ...        | ...             |         32 |             137 |
| ...         | ...        | ...             |         64 |             138 |
