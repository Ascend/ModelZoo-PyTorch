# ChatGLM2-6B


## 概述

### 简介
ChatGLM2-6B 是开源中英双语对话模型 ChatGLM-6B 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，ChatGLM2-6B 拥有更强大的性能和更长的上下文以及更高效的推理。
- 参考实现 ：
  ```
  url=https://github.com/THUDM/ChatGLM2-6B
  commitID=921d7e9adc69020a19169d1ba4f76c2675a2dd29
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
  | PyTorch 2.1 |transformers == 4.29.0; deepspeed == 0.9.2 |


- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```python
  pip install -r requirements.txt
  
  # 使用fix文件夹下的tranining_args.py替换路径下transformers/tranining_args.py
  # cp fix/utils.py /root/miniconda3/envs/conda环境名/lib/python3.7/site-packages/transformers/generation/
  ```

### 准备数据集

1. 获取数据集。

  用户可以从[这里](https://huggingface.co/datasets/shibing624/AdvertiseGen/tree/main)下载数据集，并将其放在`ptuning`路径下的`AdvertiseGen`文件夹内，该文件夹内容包括：
```
├── AdvertiseGen
      ├──train.json
      ├──dev.json
```

2. 数据转换
修改数据转换脚本`tuning/preprocess.sh`

```shell
# modify the script according to your own  ascend-toolkit path
source env_npu.sh

# for preprocess training datasets
--do_train \
--max_source_length 4096 \ #for example 
--max_target_length 4096 \  
```
```shell
# for preprocess predict datasets
--do_predict \
--max_source_length 256 \
--max_target_length 256
```
执行下面代码转换数据集

```shell
  # process datasets                              
  bash preprocess.sh
```

### 准备预训练权重
  用户可以从[这里](https://huggingface.co/THUDM/chatglm2-6b/tree/dev)下载预训练权重和配置文件，然后将这些文件放在 "model"文件夹中，**不要覆盖 `modeling_chatglm.py`文件**。
`model`文件夹内容如下：
```shell
  ├── model
      ├──config.json
      ├──configuration_chatglm.py
      ├──pytorch_model-00001-of-00007.bin
      ├──pytorch_model-00002-of-00007.bin
      ├──pytorch_model-00003-of-00007.bin
      ├──pytorch_model-00004-of-00007.bin
      ├──pytorch_model-00005-of-00007.bin
      ├──pytorch_model-00006-of-00007.bin
      ├──pytorch_model-00007-of-00007.bin
      ├──pytorch_model.bin.index.json
      ├──quantization.py
      ├──test_modeling_chatglm.py
      ├──tokenization_chatglm.py
      ├──tokenizer_config.json
      ├──tokenizer.model
      ├──modeling_chatglm.py
```


## 开始训练

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```
2. 启动训练

   该模型P-Tuning v2支持单机单卡，全参数fintune支持单机8卡。

-  全参数finetune
   配置ChatGLM2-6B训练脚本: `ptuning/ds_train_fintune.sh`

```shell
# modify the script according to your own  ascend-toolkit path
source env_npu.sh

# modify script according to your own needs
--model_name_or_path ../model/ \  #model path
--max_source_length 4096 \
--max_target_length 4096 \  #should align with the processed dataset
```
启动8卡微调

```shell
bash ds_train_fintune.sh
```

- P-Tuning v2

  启动P-Tuning v2。

  ```
  bash train.sh
  ```
3. 全参数finetune验证

    运行以下命令

    ```
    cd /${模型文件夹名称}/ptuning
    bash evaluate_fintune.sh
	```
模型训练脚本部分参数说明如下。

   ```
   --model_name_or_path                   // 模型路径
   --output_dir                           // 模型输出路径
   --gradient_accumulation_steps          // 梯度累计步长
   --learning_rate                        // 学习率
   ```

### 训练结果展示

**表 2**  训练结果展示表
| Device    | Torch_version       | total Iterations | throughput rate (samples/s) | throughput rate (tokens/s/p) | single-step time (s/step) | floating point operation (TFLOPs/s) |
| --------- | ----------- | ---------------- | ----------------------------- | ---------------------------- | ------------------------- | ----------------------------------- |
| 8p-NPU| 2.1 | 1000  | 1.79 |1833   | 4.46| 65.72 |
| 8p-竞品 | 2.1 | 1000 | 1.76 | 1802| 4.54| 64.64 |


## 推理

### 推理环境搭建
推理环境搭建参考上述训练环境搭建。

### 推理脚本

1）执行`vim infer.py`创建推理脚本，然后将下面代码写入`infer.py`文件中，然后按`Esc`键输入`:wq`退出并保存文件。

```python
from transformers import AutoTokenizer, AutoModel

# 修改CHECKPOINT路径
CHECKPOINT="./model_weight"
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModel.from_pretrained(CHECKPOINT, trust_remote_code=True, device='cuda')
model = model.eval()

print("请输入对话")
_input=input(">>")

while _input:
    response, history = model.chat(tokenizer, _input, history=[])
    print(response)
    _input=input(">>")
```
2）运行下面命令执行推理任务
```python
 python infer.py
```
### 推理结果展示
```shell
请输入对话
>>你好
你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。
>>晚上睡不着应该怎么办
以下时是一些有助于晚上睡觉的技巧:

1. 创建一个规律的睡眠时间表:每天在相同的时间上床并起床可以帮助身体建立一个规律的睡眠时间表。

2. 创建一个舒适的睡眠环境:在安静、黑暗、凉爽、舒适的房间里睡觉可以帮助放松身心,更容易入睡。

3. 避免使用电子设备:在睡觉前一两个小时内避免使用电子设备,如手机、电脑、平板电脑等,以免干扰睡眠。

4. 放松身心:在睡觉前做些轻松的活动,如阅读、听轻柔的音乐、洗个热水澡、做些瑜伽、冥想等,有助于放松身心,减轻压力。

5. 避免咖啡因和酒精:在睡觉前几个小时内避免摄入咖啡因和酒精,以免影响睡眠。

6. 远离刺激:在睡觉前远离刺激,如避免摄入咖啡因、饮酒、吸烟等,以免影响睡眠。

7. 远离压力:避免在睡觉前进行紧张的活动,如激烈的运动,以免影响睡眠。

如果这些技巧不能解决你的问题,你可以尝试寻求医生的帮助,找到更好的解决方案。
```


## 评估
### 准备数据集任务
用户可以从 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/e84444333b6d434ea7b0) 下载处理好的 C-Eval 数据集，解压到 `evaluation` 目录下。
## 运行评估任务
1）首先修改评估脚本`evaluation/evaluate_ceval.py`。
```python
# 修改 CHECKPOINT 路径和数据集任务路径
CHECKPOINT= "../model"
DATA_PATH="./CEval/test/**/*.jsonl"
```
2）然后运行下面代码执行评估任务。
```python
cd evaluation
python evaluate_ceval.py
```

### 评估结果展示

**表 3**  评估结果展示表
|任务|验证集|模型|昇腾值|参考值|社区值|
| ------------ | ------------ |  ------------ | ------------ |------------ |------- |
| CEval | test  |ChatGLM2-6B   |  0.31055 | 0.31092 |-- |



## FAQ

1. 报错提示deepspeed.py需要版本大于等于0.6.5

   ```
   # 关闭版本检测（如安装0.9.2版本无需此操作）
   # 若遇到该报错
   pip show transformers
   # 复制Location路径
   # 使用fix文件夹下的deepspeed.py替换路径下transformers/deepspeed.py
   ```

2. 加载参数阶段有卡死现象

   ```
   删除root下的cache目录，重新运行
   ```

3. 单卡阶段报embedding_dense_grad算子错误

   ```
   enbedding当前版本，不支持动静合一，静态有部分shape不支持,新版本已修复
   # 若遇到该报错
   修改main.py文件
   torch.npu.set_compile_mode(jit_compile=False)
   ```

4. 提示so文件错误

   ```
   提示so文件找不到
   # 若遇到该报错
   全局搜索so的位置，然后导入环境变量
   export LD_LIBRARY_PATH=/usr/:$LD_LIBRARY_PATH
   ```

5. eval提示scaledsoftmax报错

   ```
   算子shape泛化性还有问题
   # 若遇到该报错
   搜索output文件夹生成的modeling_chatglm.py文件，
   self.scale_mask_softmax 设置为false
   ```

```
* 规避推理错误：

`cp fix/utils.py /root/miniconda3/envs/conda环境名/lib/python3.7/site-packages/transformers/generation/`
```

1. 微调时出现AttributeError或RuntimeError

   module 'torch_npu' has no attribute 'npu_rotary_mul' 或

   RuntimeError:Error!, The last dimension of input tensor shoule be within the range of [32,2048] and be divisible by32

   ```
   修改modeling_chatglm.py文件:
   USE_NPU_ROTARY=False
   USE_SCALED_SOFTMAX=False
   ```

   PS: 设置为True能提高性能

2. 如果cann不支持flash_attention

    报错提示为module 'torch_npu' has no attribute 'npu_flash_attention'

```
修改modeling_chatglm.py文件:
USE_FLASH=False
```

​       PS: 设置为True能提高性能


* 规避评估错误：


1. 任务评估过程中出现`XXXX does not appear to have a file named tokenization_chatglm.py/ configuration_chatglm.py / modeling_chatglm.py`时，将`model`路径下的`tokenization_chatglm.py/ configuration_chatglm.py / modeling_chatglm.py`拷贝到`XXXX`路径下。
2. 任务评估过程中如果报错`RuntimeError: call aclnnFlashAttentionScore failed`，请修改`XXXX`路径下的`modeling_chatglm.py`。
   ```python
   # 修改modeling_chatglm.py文件:
   USE_FLASH=False
   ```


## 公网地址说明

代码涉及公网地址参考 public_address_statement.md