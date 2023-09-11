# LLaMA-13B模型-推理指导

- [概述](#概述)

- [输入输出数据](#输入输出数据)

- [推理环境准备](#推理环境准备)

- [快速上手](#快速上手)

  - [获取源码及依赖](#获取源码及依赖)                                                                                                          
  - [模型推理](#模型推理)

- [模型推理性能](#模型推理性能)

# 概述

   LLaMA（Large Language Model Meta AI），由 Meta AI 发布的一个开放且高效的大型基础语，可以通过自然语言交互的方式提供知识、文本生成、语言翻译、语言理解、代码编写和解释等任务。

- 参考实现：
   ```bash
   https://github.com/facebookresearch/llama
   ```

# 输入输出数据
- 输入数据

  | 输入数据      | 大小          | 数据类型  | 数据排布格式 | 是否必选 |
  |-----------|-------------|-------|--------|------|
  | input_ids | BATCH_SIZE x SEQ_LEN | INT64 | ND     | 是    |
  | attention_mask | BATCH_SIZE x 1 x SEQ_LEN x SEQ_LEN | FLOAT32 | ND     | 否|

- 输出数据

  | 输出数据       | 大小                 | 数据类型  | 数据排布格式 |
  |------------|--------------------|-------|--------|
  | output_ids | BATCH_SIZE x OUTPUT_SEQ_LEN | INT64 | ND     |


# 推理环境准备

 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套             | 版本       | 下载链接                                                                                                                                                                                 |
  |----------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | 固件与驱动          | 23.0.T50 | -    |
  | CANN           | 7.0.T3   | - |
  | Python         | 3.7.5    | -                                                                                                                                                                                 |         
  | PytorchAdapter | 1.11.0   | -      |
  | 推理引擎           | -   |     -     |

  **表 2** 推理引擎依赖

   | 软件     | 版本要求      | 
   |--------|-----------|
   | glibc  | \>= 2.27  | 
   | gcc    | \>= 7.5.0 | 

  **表 3** 硬件形态

   | CPU    | Device   |
   |--------|----------|
   | aarch64 | 300I DUO |

# 快速上手

## 获取源码及依赖

1. 环境部署
- 安装HDK
- 安装CANN
- 安装PytorchAdapter
- 安装依赖   
   参考[推理环境准备](#推理环境准备)安装配套软件。安装python依赖。
   ```bash
   pip3 install -r requirements.txt
   ```

2. 下载LLaMA-13B模型权重，放置到自定义`input_dir`
   ```bash
   https://huggingface.co/decapoda-research/llama-13b-hf
   ```

3. 安装加速库
   下载推理引擎文件，`ascend_acclib.zip`
   ```bash
   # 解压
   unzip ascend_acclib.zip
   cd ascend_acclib
   source set_env.sh
   ```

## 模型推理

1. 切分模型权重
首次跑模型时，需要先对模型权重进行切分，切分方法如下
- 修改代码
   1. 拷贝`modeling_llama_parallel_cut.py`到对应的transformer库路径下，示例：
      ```bash
      cp modeling_llama_parallel_cut.py /usr/local/python3.7/site-packages/transformers/models/llama/modeling_llama.py
      ```
   2. 修改`cut_model_and_run_llama.sh`中`input_dir`为真实`input_dir`
   3. 修改`cut_model_and_run_llama.sh`中`output_dir`为自定义路径，用于存放切分后的模型权重
- 执行切分
   ```bash
   bash cut_model_and_run_llama.sh
   # 切分好的模型权重会存放在自定义的output_dir
   ```

2. 执行模型推理
模型切分完成后，cut_model_and_run_llama.sh会加载`output_idr`下切分好的模型权重（`output_dir/part_model/0`和`output_dir/part_model/1`）进行推理
- 修改代码
   拷贝`modeling_llama_ascend.py`到对应的transformer库路径下（示例：`python3.7/site-packages/transformers/models/llama/modeling_llama.py`）
  ```bash
  cp modeling_llama_ascend.py /usr/local/python3.7/site-packages/transformers/models/llama/modeling_llama.py
  ```

- 执行推理
   ```bash
   bash cut_model_and_run_llama.sh
   ```
   该命令会运行一次简单的推理实例warm up，并启动后续的16个问答
- 自定义运行可参考`run_llama1_13b_parallel.py`

# 模型推理性能

| 硬件形态  |   输入长度   |  输出长度  |     解码速度      |
|:-----:|:--------:|:------:|:-------------:|
| Duo双芯 | 1 x 1024 |   1024   | 6.32 tokens/s |



