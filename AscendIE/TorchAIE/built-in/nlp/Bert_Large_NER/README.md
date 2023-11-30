- [1. 概述](#1-概述)
  - [1.1. 输入输出数据](#11-输入输出数据)
- [2. 推理环境准备](#2-推理环境准备)
- [3. 快速上手](#3-快速上手)
  - [3.1. 获取模型](#31-获取模型)
  - [3.2. 准备数据集(请遵循数据集提供方要求使用)](#32-准备数据集请遵循数据集提供方要求使用)
  - [3.3. 模型推理](#33-模型推理)
- [4. 模型推理性能\&精度](#4-模型推理性能精度)



# 1. 概述

**bert-large-NER**  是一个经过微调的 BERT 模型，可用于**命名实体识别**，并为 NER 任务实现**一流的性能**。它已经过训练，可以识别四种类型的实体：位置（LOC），组织（ORG），人员（PER）和杂项（杂项）。

具体而言，此模型是一个*bert-large-cased*模型，在标准  [CoNLL-2003 命名实体识别](https://www.aclweb.org/anthology/W03-0419.pdf)数据集的英文版上进行了微调。如果要在同一数据集上使用较小的 BERT 模型进行微调，也可以使用[**基于 NER 的 BERT**](https://huggingface.co/dslim/bert-base-NER/)  版本。

* 参考实现：

  ```
  url = https://huggingface.co/dslim/bert-large-NER
  commit_id =  95c62bc0d4109bd97d0578e5ff482e6b84c2b8b9
  model_name = bert-large-NER
  ```

## 1.1. 输入输出数据

- 输入数据

  | 输入数据       | 数据类型 | 大小            | 数据排布格式 |
  | -------------- | -------- | --------------- | ------------ |
  | input_ids      | int64    | batchsize x 512 | ND           |
  | attention_mask | int64    | batchsize x 512 | ND           |
  | token_type_ids | int64    | batchsize x 512 | ND           |


- 输出数据

  | 输出数据 | 大小                 | 数据类型 | 数据排布格式 |
  | -------- | -------------------- | -------- | ------------ |
  | logits   | batchsize x 512  x 9 | FLOAT32  | ND           |


# 2. 推理环境准备

- 硬件环境：310P3
  

-   该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套       | 版本             | 环境准备指导 |
| ---------- | ---------------- | ------------ |
| 固件与驱动 | 23.0.rc1         | -            |
| CANN       | 7.0.RC1.alpha003 | -            |
| Python     | 3.9.11           | -            |
| Pytorch    | 2.0.1+cpu        | -            |
| AscendIE   | 6.3.RC2          | -            |
| Torch-aie  | 6.3.RC2          | -            |



# 3. 快速上手

## 3.1. 获取模型

1.  获取开源模型。
     1. 获取模型权重和配置文件
        ```
        git clone https://huggingface.co/dslim/bert-large-NER
        ```
      得到如下文件：
      ```
      bert-large-NER/
      ├── config.json
      ├── dslim_bert-large-NER #U00b7 Hugging Face_files
      │   ├── 1655075923870-5e7565183d77a72421292d00.png
      │   ├── analytics.js.#U4e0b#U8f7d
      │   ├── css2
      │   ├── css2(1)
      │   ├── huggingface_logo-noborder.svg
      │   ├── inner.html
      │   ├── js
      │   ├── katex.min.css
      │   ├── m-outer-27c67c0d52761104439bb051c7856ab1.html
      │   ├── m-outer-6576085ca35ee42f2f484cda6763e4aa.js.#U4e0b#U8f7d
      │   ├── out-4.5.43.js.#U4e0b#U8f7d
      │   ├── saved_resource
      │   ├── script.js.#U4e0b#U8f7d
      │   └── style.css
      ├── dslim_bert-large-NER #U00b7 Hugging Face.html
      ├── pytorch_model.bin
      ├── README.md
      ├── special_tokens_map.json
      ├── tokenizer_config.json
      └── vocab.txt
      ```
     模型文件只需要下载`pytorch_model.bin`即可。

2.  安装依赖。
  
  ```
  pip install -r requirements.txt
  ```
  
## 3.2. 准备数据集(请遵循数据集提供方要求使用)

1.  获取原始数据集。

   数据集名称： **CoNll-2003**：信息抽取-命名实体识别（NER）
   下载链接：[CoNLL 2003 (English) Dataset | DeepAI](https://data.deepai.org/conll2003.zip)
   解压:
   ```
   unzip conll2003.zip
   ```
   得到如下文件夹：
   ```
   conll2003
   ├── metadata
   ├── train.txt
   ├── test.txt
   └── valid.txt
   ```

## 3.3. 模型推理                                                                                
1. 导出torch script模型：
  ```
  python3 export_trace_model.py
  ```
  得到导出后的ts模型：`bert_large_ner.pt`

2. 修改`bert_large_ner.pt`
  > 【注意】 因为`bert_large_ner`模型的`attention_mask`计算部分的`mul`算子的第二个初始化入参`CONSTANTS.c0)`被初始化为`float`数据类型的最小值，
  > 该值超出了`fp16`数据类型能够表示的最小值的范围，出现了下溢，如果不做下面的模型修改，将导致模型的精度下降(经测acc=76%)。
  > 经过下面的修改后，模型精度可以达到87.43%， 与om推理的版本仍然存在一定差距，该问题定位当中。

   1. 解压 `bert_large_ner.pt`
   ```
   unzip -q bert_large_ner.pt
   ```
   得到`bert_large_ner`文件夹
   2. 修改`bert_large_ner/code/__torch__/transformers/models/bert/modeling_bert.py`文件的第36行，
   修改前：
   ```
   attention_mask0 = torch.mul(torch.rsub(_4, 1.), CONSTANTS.c0)
   ```
   修改后：
   ```
   attention_mask0 = torch.mul(torch.rsub(_4, 1.), -1000.0)
   ```
   保存修改内容。
   3. 重新压缩，得到修改后的`bert_large_ner.pt`文件
   ```
    zip -r -q bert_large_ner.pt bert_large_ner/
   ```


3. 模型推理
   ```
   python3 run_torch_aie.py
   ```

# 4. 模型推理性能&精度

1. 性能对比

| Batch Size | om推理（onnx改图匹配bert大kernel融合算子） | torch-aie推理 | torch-aie/om |
| :--------: | :----------------------------------------: | :-----------: | :----------: |
|     1      |                63.3062 it/s                |  39.69 it/s   |    0.6269    |

> 性能有改进空间，待通过aie接入bert优化pass。

1. 精度对比

|      模型      | Batch Size | om推理 | torch-aie推理 |
| :------------: | :--------: | :----: | :-----------: |
| bert_large_NER |     1      | 90.74% |    87.43%     |

> 原始模型在Pytorch CPU 框架下测试的精度为87.43，与PT插件精度一致。
