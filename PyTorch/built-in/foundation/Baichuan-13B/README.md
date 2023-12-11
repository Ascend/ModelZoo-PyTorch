#  baichuan-13B

## 概述

### 简介

LLaMA Factory是一个易于使用的LLM微调框架。它使用一个简单的Web界面。用户仅需10分钟即可完成模型的自感知微调。支持LLAMA、Falcon等多种主流开源模型。此外，还提供了语言、模型路径等自定义选项。训练完成后，您可以评估模型效果，并将模型导出给其他系统使用。

- 参考实现 ：

  ```
  url=https://github.com/hiyouga/LLaMA-Factory/commits/v0.2.0
  commit_id=7a5318804870b1f2bedec8d4a676e465b48d5c3e
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/foundation
  ```

## 准备训练环境

### 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 2.0 |transformers == 4.31.0；accelerate==0.21.0|


- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```python
  pip install -r requirements.txt
  
  # 使用项目`utils`目录下的`train_bash.py`文件替换`./${模型文件夹名称}/src`路径下的`train_bash.py`.
  # 使用项目`utils`目录下的`misc.py`文件替换`./${模型文件夹名称}/src/llmtuner/extras`路径下的`misc.py`.
  # 使用项目`utils`目录下的`modeling_baichuan.py`文件替换`./${模型文件夹名称}/model_weight`路径下的`modeling_baichuan.py`.
  ```

### 准备数据集

  项目的"./data"路径下已存在预训练所需数据集。

```shell
data/
├── alpaca_data_en_52k.json
├── alpaca_data_zh_51k.json
├── alpaca_gpt4_data_en.json
├── alpaca_gpt4_data_zh.json
├── belle_multiturn
│   └── belle_multiturn.py
├── comparison_gpt4_data_en.json
├── comparison_gpt4_data_zh.json
├── dataset_info.json
├── example_dataset
│   ├── example_dataset.py
│   └── examples.json
├── hh_rlhf_en
│   └── hh_rlhf_en.py
├── lima.json
├── oaast_rm.json
├── oaast_rm_zh.json
├── oaast_sft.json
├── oaast_sft_zh.json
├── README.md
├── README_zh.md
├── self_cognition.json
├── sharegpt_zh_27k.json
├── ultra_chat
│   └── ultra_chat.py
└── wiki_demo.txt
```

### 准备预训练权重
  用户从[链接](https://huggingface.co/baichuan-inc/Baichuan-13B-Base/tree/main) 自行获取模型配置文件和权重文件，并放于model 目录下，微调依赖该模型权重，文件夹内容如下：
```shell
├──model
    ├── config.json
    ├── configuration_baichuan.py
    ├── generation_config.json
    ├── modeling_baichuan_origin.py
    ├── modeling_baichuan.py
    ├── pytorch_model-00001-of-00003.bin
    ├── pytorch_model-00002-of-00003.bin
    ├── pytorch_model-00003-of-00003.bin
    ├── pytorch_model.bin.index.json
    ├── quantizer.py
    ├── requirements.txt
    ├── special_tokens_map.json
    ├── tokenization_baichuan.py
    ├── tokenizer_config.json
    └── tokenizer.model
```


### 配置双机通信环境

1.安装pdsh
url： https://github.com/chaos/pdsh/tree/pdsh-2.29


**安装**
```python
git clone https://github.com/chaos/pdsh/archive/refs/tags/pdsh-2.29.tar.gz

tar -zxvf pdsh-2.29.tar.gz
cd pdsh-2.29
./configure --with-ssh --with-rsh --with-mrsh --with-mqshel --with-qshell  --with-dshgroups --with-machines=/etc/pdsh/machines  --without-pam

make
make install
```
安装完成后，执行`pdsh -h`命令。显示如下信息，表示安装成功。
```shell
# pdsh -h
Usage: pdsh [-options] command ...
-S                return largest of remote command return values
-h                output usage menu and quit
-V                output version information and quit
-q                list the option settings and quit
-b                disable ^C status feature (batch mode)
-d                enable extra debug information from ^C status
-l user           execute remote commands as user
-t seconds        set connect timeout (default is 10 sec)
-u seconds        set command timeout (no default)
-f n              use fanout of n nodes
-w host,host,...  set target node list on command line
-x host,host,...  set node exclusion list on command line
-R name           set rcmd module to name
-M name,...       select one or more misc modules to initialize first
-N                disable hostname: labels on output lines
-L                list info on all loaded modules and exit
-g groupname      target hosts in dsh group "groupname"
-X groupname      exclude hosts in dsh group "groupname"
-a                target all nodes
available rcmd modules: ssh,rsh,exec (default: rsh)

```

2.双机通信配置

首先，我们需要编辑两台服务器的/etc/hosts文件，添加两台服务器的IP地址，并将ip1和ip2替换为两台服务器的实际IP地址

```shell
vim /etc/hosts
```
```shell
ip1 node1
ip2 node2
```

然后，我们需要执行以下命令来生成sshkey。

```shell
ssh-keygen -t rsa
```
接着，将ssh-key拷贝到每个节点，本机也要拷贝。

```shell
ssh-copy-id root@ip1
ssh-copy-id root@ip2
```
然后，在每个节点上运行以下代码。如果不需要输入密码，则表示配置成功。然后执行`exit`退出。

```shell
ssh root@ip1
ssh root@ip2
```

## 开始训练 


1、将项目根目录下的`run_baichuan_sft_2m.sh`、`ds_config_zero2.json`、`hostfile`文件拷贝到`${模型文件夹名称}`路径下。
```shell
cp ../run_baichuan_sft_2m.sh .
cp ../ds_config_zero2.json .
cp ../hostfile .
```

2、启动脚本
该模型双机16卡微调，执行如下命令启动训练。
```shell
sh run_baichuan_sft_2m.sh
``` 
模型训练部分参数说明如下：

   ```
--deepspeed                     //使用DeepSpeed分布式训练框架。
--dataset                       //指定训练数据集。
--finetuning_type               //指定微调类型。
--output_dir                    //指定输出目录。
--per_device_train_batch_size   //每个设备的训练批次大小。
--gradient_accumulation_steps   //梯度累积步数。
--learning_rate                 //学习率。
--num_train_epochs              //训练的轮数。
--fp16                          //使用fp16精度浮点数进行训练。

   ```
   **注**：为确保双机训练成功，请保证双机环境及路径一致，包括项目路径、conda环境、cann和驱动等。
训练完成后，权重文件保存`--output_dir`参数指定的路径下，并输出模型训练相关信息。

## 训练结果展示

**表 2**  训练结果展示表
| Device |Torch_Version   |  total epochs | train loss | train samples per second | train steps per second| 
| --------- | ----- | ----------- | ----------------------------- | ---------------------------- | ------------------------- | 
| 16p-NPUs|2.0.1      |  10.0  | 0.903  |  11.378     |   0.022     | 
| 16p-竞品 | 2.0.1| 10.0  | 0.903  |  9.3     |     0.018   | 


## 推理

### 推理环境搭建
1. 推理环境搭建参考上述训练环境搭建。
2. 准备推理权重。
用户从[链接](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/tree/main) 自行获取模型配置文件和权重文件，并放于Baichuan-13B-Chat 目录下。
```shell
├── config.json
├── configuration_baichuan.py
├── generation_config.json
├── generation_utils.py
├── handler.py
├── hf_pytorch_model.bin.index.json
├── modeling_baichuan.py
├── pytorch_model-00001-of-00003.bin
├── pytorch_model-00002-of-00003.bin
├── pytorch_model-00003-of-00003.bin
├── pytorch_model.bin.index.json
├── quantizer.py
├── README.md
├── requirements.txt
├── special_tokens_map.json
├── tokenization_baichuan.py
├── tokenizer_config.json
└── tokenizer.model

```

### 推理脚本

1）执行`vim infer.py`创建推理脚本，然后将下面代码写入`infer.py`文件中，然后按`Esc`键输入`:wq`退出并保存文件。

```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

model_weight_path = 'Baichuan-13B-Chat/'
tokenizer = AutoTokenizer.from_pretrained(model_weight_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_weight_path, device_map="npu:1", torch_dtype=torch.float16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(model_weight_path)
messages = []
messages.append({"role": "user", "content":"解释一下“温故而知新”" })
response = model.chat(tokenizer, messages)

print(response)
```

infer.py文件中的配置参数:
```python
# 指定加载的模型权重为上述下载的权重和配置文件夹。
model_weight_path = 'Baichuan-13B-Chat/'     # 模型权重
device_map="npu:1"                           # 指定运行的NPU卡
``` 
2）运行下面命令执行推理任务
```python
 python infer.py
```
## 推理结果展示
```shell
嘃 温故而知新,可以为师矣。 解释:温习学过的知识,从而得到新的理解和体会。也指回忆过去,能更好地认识现在。 温故而知新,可以为师矣。 解释:复习旧的知识,能够从中有新的收获。这样的人就可以做老师了。 温故而知新,可以为师矣。 解释:复习旧的知识,能够从中有新的收获。这样的人就可以做老师了。
```


## 评估

### 准备数据集任务
在项目的`evaluation` 目录下已经存在评估任务数据集：
```shell
evaluation
├── ceval
│   ├── ceval.py
│   ├── ceval.zip
│   └── mapping.json
├── cmmlu
│   ├── cmmlu.py
│   ├── cmmlu.zip
│   └── mapping.json
└── mmlu
    ├── mapping.json
    ├── mmlu.py
    └── mmlu.zip

```

### 运行评估任务

1. 执行`vim evaluation.sh`创建推理脚本，然后将下面代码写入`evaluation.sh`文件中，然后按`Esc`键输入`:wq`退出并保存文件。
```shell
#!/bin/bash

MODEL_NAME_OR_PATH=./model_weight
CHECKPOINT=./model_weight


ASCEND_RT_VISIBLE_DEVICES=1 python src/evaluate.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --finetuning_type full \
    --checkpoint_dir $CHECKPOINT \
    --template default \
    --task ceval \
    --split validation \
    --lang en \
    --n_shot 5 \
    --batch_size 4
```
2. 然后运行下面代码执行评估任务。
```python
bash evaluation.sh
```


### 评估结果展示

  **表 3**  评估结果展示表
|任务|模型|昇腾值|参考值|社区值|
| ------------ |  ------------ | ------------ |------------ |------- |
| CEval |Baichuan-13B   |  43.98 | 42.72 |-- |


## FAQ

**为适配V0.2.0的代码，在配置完运行环境后做如下修改：**

1、检测下面python包并安装对应版本。
```python
pip install trl==0.7.2
pip install transformers==4.31.0
pip install transformers_stream_generator decorator absl-py cloudpickle synr==0.5.0 tornado
```

2、修改deepspeed版本检测。

  - 注释 `${conda环境路径}/lib/python3.8/site-packages/transformers/deepspeed.py` line65的deepspeed版本检测代码。

  - 将 `${conda环境路径}/lib/python3.8/site-packages/accelerate/accelerator.py` line296修改为`if compare_versions("deepspeed", "<", "0.9.2"):`



## 引用

```
@Misc{llama-factory,
  title = {LLaMA Factory},
  author = {hiyouga},
  howpublished = {\url{https://github.com/hiyouga/LLaMA-Factory}},
  year = {2023}
}
```