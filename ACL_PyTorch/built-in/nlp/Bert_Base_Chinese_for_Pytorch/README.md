# Bert_Base_Chinese模型-推理指导

## 概述

`BERT`来自 Google 的论文`Pre-training of Deep Bidirectional Transformers for Language Understanding`，`BERT` 是`Bidirectional Encoder Representations from Transformers`的首字母缩写，整体是一个自编码语言模型。`Bert_Base_Chinese`是`BERT`模型在中文语料上训练得到的模型。

+ 参考实现：

  ```shell
  url=https://huggingface.co/bert-base-chinese
  commit_id=38fda776740d17609554e879e3ac7b9837bdb5ee
  mode_name=Bert_Base_Chinese
  ```

  通过Git获取对应的commit_id的代码方法如下：

  ```shell
  git clone https://gitee.com/ascend/ModelZoo-PyTorch
  cd ModelZoo-PyTorch
  git checkout master
  cd ACL_PyTorch/built-in/nlp/Bert_Base_Chinese_for_Pytorch
  ```


## 输入输出数据

+ 输入数据

  | 输入数据       | 数据类型 | 大小                   | 数据排布格式 |
  | -------------- | -------- | ---------------------- | ------------ |
  | input_ids      | FP32     | batchsize x seq_length | ND           |
  | attention_mask | FP32     | batchsize x seq_length | ND           |
  | token_type_ids | FP32     | batchsize x seq_length | ND           |

+ 输出数据

  | 输出数据 | 数据类型 | 大小                   | 数据排布格式 |
  | -------- | -------- | ---------------------- | ------------ |
  | out      | FP32     | batchsize x seq_length | ND           |


## 推理环境准备

+ 表1 版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 5.1.RC2 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 |                                                              |
  | Python                                                       | 3.7+    |                                                              |
  | PyTorch                                                      | 1.8.0+  |                                                              |
  | 说明：Altlas 300I Duo推理卡请以CANN版本选择实际固件于驱动版本 | \       | \                                                            |


## 快速上手

1. 安装依赖（建议按照自己需求安装）

   基础依赖：

   ```shell
   pip3 install -r requirement.txt
   ```

   安装改图工具依赖：

   ```shell
   git clone https://gitee.com/Ronnie_zheng/MagicONNX.git MagicONNX
   cd MagicONNX && git checkout cb071bb62f34bfae405af52063d7a2a4b101358a
   pip3 install . && cd ..
   ```

## 准备数据集

1. 获取原始模型

   ```shell
   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/bert-base-chinese
   ```

   

2. 获取原始数据集

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

3. 数据预处理

   ```shell
   # 输入参数：${input_patgh} ${model_dir} ${save_dir} ${seq_length}
   python3 preprocess.py ./zhwiki-latest-pages-articles_validation.txt ./bert-base-chinese ./input_data/ 384
   ```


## 模型推理

1. 模型转换

   获取权重文件：[pytorch_model.bin](https://huggingface.co/bert-base-chinese/blob/main/pytorch_model.bin)，替换`bert-base-chinese`目录下的文件：

   ```shell
   mv pytorch_model.bin bert-base-chinese
   ```

2. 导出onnx文件

   ```shell
   # 输入参数：${model_dir} ${output_path} ${seq_length} 
   python3 pth2onnx.py ./bert-base-chinese ./bert_base_chinese.onnx 384
   # 修改优化模型：${bs}:[1, 4, 8, 16, 32, 64],${seq_len}:384
   python3 -m onnxsim ./bert_base_chinese.onnx ./bert_base_chinese_bs${bs}.onnx --input-shape "input_ids:${bs},${seq_len}" "attention_mask:${bs},${seq_len}" "token_type_ids:${bs},${seq_len}"
   python3 fix_onnx.py bert_base_chinese_bs${bs}.onnx bert_base_chinese_bs${bs}_fix.onnx
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
      atc --model=./bert_base_chinese_bs${bs}_fix.onnx --framework=5 --output=./bert_base_chinese_bs${bs} --input_format=ND --log=debug --soc_version=${chip_name} --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance
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

4. 安装ais_bench推理工具

   请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

5. 执行推理

   ```shell
   # 以bs1模型推理为例
   mkdir -p ./output_data/bs1
   python3 -m ais_bench --model ./bert_base_chinese_bs1.om --input ./input_data/input_ids,./input_data/attention_mask,./input_data/token_type_ids --output ./output_data/bs1 --batchsize 1 --device 1
   ```

6. 精度验证

   ```shell
   # 以bs1模型推理为例
   # 输入参数：${result_dir} ${gt_dir} ${seq_length}
   python3 postprocess.py ./output_data/bs${bs}/${timestamp} ./input_data/labels 384
   ```

## 模型推理性能&精度

调用ACL接口推理计算，性能参考下列数据：

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

## 其他下游任务

+ [序列标注(Sequence Labeling)](downstream_tasks/sequence_labeling/README.md)

