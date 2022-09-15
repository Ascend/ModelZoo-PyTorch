# Bert_Base_Chinese模型-推理指导

## 概述

`BERT`来自 Google 的论文`Pre-training of Deep Bidirectional Transformers for Language Understanding`，`BERT` 是`Bidirectional Encoder Representations from Transformers`的首字母缩写，整体是一个自编码语言模型。`Bert_Base_Chinese`是`BERT`模型在中文语料上训练得到的模型。

`sequence_labeling`: 基于BERT+CRF的序列标注子任务

+ 参考实现：

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


  通过Git获取对应的commit_id的代码方法如下：

  ```shell
  git clone https://gitee.com/ascend/ModelZoo-PyTorch
  cd ModelZoo-PyTorch
  git checkout master
  cd ACL_PyTorch/built-in/nlp/Bert_Base_Chinese_for_Pytorch/downstream_tasks/sequence_labeling
  ```


## 输入输出数据

+ 输入数据

  | 输入数据       | 数据类型 | 大小                   | 数据排布格式 |
  | -------------- | -------- | ---------------------- | ------------ |
  | token_ids      | INT64    | batchsize x seq_length | ND           |

+ 输出数据

  | 输出数据       | 数据类型 | 大小                                     | 数据排布格式 |
  | --------       | -------- | ---------------------------------------- | ------------ |
  | emission_score | FP16     | batchsize x seq_length x length_category | ND           |
  | attention_mask | FP16     | batchsize x seq_length                   | ND           |


## 推理环境准备

+ 表1 版本配套表

  | 配套                                                          | 版本    | 环境准备指导                                                                                          |
  | ------------------------------------------------------------  | ------- | ------------------------------------------------------------                                          |
  | 固件与驱动                                                    | 6.1.RC2 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                          | 6.1.RC2 |                                                                                                       |
  | Python                                                        | 3.7+    |                                                                                                       |
  | PyTorch                                                       | 1.12.1  |                                                                                                       |
  | 说明：Altlas 300I Duo推理卡请以CANN版本选择实际固件于驱动版本 | \       | \                                                                                                     |


## 快速上手

1. 安装依赖（建议按照自己需求安装）

   基础依赖：

   ```shell
   pip3 install -r requirement.txt
   ```

   安装其他依赖：

   ```shell
   # 改图
   git clone https://gitee.com/Ronnie_zheng/MagicONNX.git MagicONNX
   cd MagicONNX && git checkout 99a713801fe70db702e3903744d2a6372a975fea
   pip3 install . && cd ..
   # 模型依赖
   git clone https://github.com/Tongjilibo/bert4torch.git
   cd bert4torch && git checkout c348349a4c7579d14c393ea61e868652801293ca
   pip3 install . && cd ..
   git clone https://huggingface.co/bert-base-chinese
   ```

## 准备数据集

1. 获取原始数据集

   下载china-people-daily数据：

   ```
   wget https://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
   ```

   解压得到数据文件：

   ```
   tar -xf china-people-daily-ner-corpus.tar.gz
   ```

2. 数据预处理

   ```shell
   python3 preprocess.py --input-path ./china-people-daily-ner-corpus/example.dev --out_dir ./preprocessed_data --dict_path ./bert-base-chinese/vocab.txt
   ```


## 模型推理

1. 模型转换

   获取权重文件：[best_model.pt](https://pan.baidu.com/s/1-cQ3hpB-SmB94NqwO5_7Dw), 提取码：rasv。


2. 导出onnx文件

   ```shell
   python3 pth2onnx.py --input_path best_model.pt --out_path ./models/onnx/bert_base_chinese_sequence_labeling.onnx
   # 修改优化模型：以bs64为例
   python3 -m onnxsim ./models/onnx/bert_base_chinese_sequence_labeling.onnx ./models/onnx/bert_base_chinese_bs64.onnx --input-shape "token_ids:64,256"
   python3 fix_onnx.py ./models/onnx/bert_base_chinese_bs64.onnx ./models/onnx/bert_base_chinese_bs64_fix.onnx
   ```

3. 使用ATC工具将ONNX模型转OM模型

   1. 配置环境变量

      ```shell
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      ```

   2. 执行命令查看芯片名称(${chip_name})

      ```shell
      npu-smi info  # 通过 Name Device列获取${chip_name}
      ```

   3. 执行ATC命令

      ```shell
      # bs:[1, 4, 8, 16, 32, 64]
      atc --model=./models/onnx/bert_base_chinese_bs${bs}_fix.onnx --framework=5 --output=./models/om/bert_base_chinese_bs${bs} --input_format=ND --log=debug --soc_version=${chip_name} --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance
      ```

      + 参数说明：

        + --model ：为ONNX模型文件
        + --framework：5代表ONNX模型
        + --output：输出的OM模型
        + --input_format：输入数据的格式
        + --input_shape：输入数据的shape
        + --log：日志级别
        + --soc_version：处理器型号
        + --optypelist_for_implmode：需要选择implmode的算子
        + --op_select_implmode：特定算子的实现方式

        运行成功后生成OM模型文件。

4. 使用ais-infer工具进行推理

   安装过程可参考：[ais_infer](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)

5. 执行推理

   ```shell
   # 以bs64模型推理为例
   mkdir -p ./output_data/bs64
   python3 ais_infer.py --model ./models/om/bert_base_chinese_bs64.om --input ./preprocessed_data/input_data --output ./output_data/bs64 --batchsize 64 --device 0
   ```

6. 精度验证

   ```shell
   # 以bs64模型推理为例
   python3 postprocess.py --result_dir output/bs64/${timestamp} --out_path eval.json --label_dir preprocessed_data/label
   ```

## 模型推理性能&精度

调用ACL接口推理计算，性能参考下列数据：

精度：

| 模型              | Pth精度(val-token level)                   | Pth精度(val-entity level)                  |
| :---------------: | :--------:                                 | :-------------:                            |
| Bert-Base-Chinese | f1:0.9724 precision: 0.9684 recall: 0.9765 | f1:0.9600 precision: 0.9569 recall: 0.9632 |
| 模型              | NPU精度(val-token level)                   | NPU精度(val-entity level)                  |
| :---------------: | :--------:                                 | :-------------:                            |
| Bert-Base-Chinese | f1:0.9720 precision: 0.9669 recall: 0.9772 | f1:0.9594 precision: 0.9559 recall: 0.9630 |

性能：

| 模型              | BatchSize | NPU性能 | 基准性能  |
| :---------------: | :-------: | :-----: | :-------: |
| Bert-Base-Chinese |         1 |         |           |
| Bert-Base-Chinese |         4 |         |           |
| Bert-Base-Chinese |         8 |         |           |
| Bert-Base-Chinese |        16 |         |           |
| Bert-Base-Chinese |        32 |         |           |
| Bert-Base-Chinese |        64 |         |           |
