# GPT-NeoX

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

GPT-NeoX-20B 是由EleutherAI和Hugging face合作开发的一个超大规模的语言模型，它采用了分布式训练和轻量级架构等技术，同时也有很高的精度和效率。

- 参考实现：

  ```
  url=https://github.com/EleutherAI/gpt-neox/tree/v2.0
  commit_id=9610391ab319403cef079b438edd016a2443af54
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/foundation/GPT-NeoX
  code_path=PyTorch/built-in/foundation/GPT-NeoX
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                        |
  | --------- | ------------------------------------------------------------ |
  | 固件与驱动 | 23.0.RC2 |
  | CANN       | 6.1.RC3 |
  | PyTorch    | PyTorch 1.11|
  | Python     |Python 3.7.5|



- 前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 2**  版本支持表

  | Torch_Version      |     三方库依赖版本     |
  |:---------------:| :----------------------------------------------------------: |
  | PyTorch 1.11 | deepspeed 0.9.2 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

1. 安装基础依赖

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  cd requirements
  pip install -r requirements.txt  # PyTorch1.11版本
  ```
2. 安装deepspeed_npu插件

  ```
  # adaptor分支
  git clone https://gitee.com/ascend/DeepSpeed.git
  cd Deepspeed
  pip install ./
  ```
3. 安装pdsh插件

  ```
  获取pdsh-2.34源码并解压 https://github.com/chaos/pdsh/releases/tag/pdsh-2.34
  cd pdsh-2.34
  ./configure --with-ssh
  make && make install
  ```
  



## 准备数据集

1. 获取数据集。
    ```
   方法1：
    原始数据(480G)：https://github.com/EleutherAI/the-pile
    下载源代码：git clone https://github.com/EleutherAI/the-pile.git
    下载数据集：1、进入the-pile目录；2、pip install -e；3、python the_pile/pile.py --interleave_output 30 --using pile_reprod
    下载完共30个文件、480G；单个文件15G、解压后43G；文件命名分别为：00.jsonl.zst~29.jsonl.zst
   方法2：
    数据集：https://opendatalab.com/  可从此链接直接下载压缩后的数据集，需要下载的数据集参考方法1中的原始数据集
    解压工具：https://github.com/facebook/zstd
    解析命令：1、tar -zxvf zstd-1.5.5.tar.gz；2、进入zstd-1.5.5目录执行：make && make install；3、解压缩.zst文件：zstd -d *.zst
   ```
   
2. 词表
   ```
   词表：
   GPT2BPETokenizer: GPT2 tokenizer vocab, and merge files from the following links:
   Vocab: https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
   Merge: https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
   
   HFTokenizer: use the 20B tokenizer (for which only a single Vocab file is needed):
   Vocab: https://the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/20B_tokenizer.json
    
   ```
3. 数据预处理（按需处理所需要的数据集）

   ```
   Demo示例参考：https://gitee.com/l30040116/gpt-neo-x-data_process
   
   依赖包：ujson、lm-dataformat、ftfy
   source_path="定义数据路径"
   out_path="输出路径"
   word_path="词表路径"
   source_data=" $source_path/00.jsonl,$source_path/01.jsonl,$source_path/02.jsonl,$source_path/03.jsonl,$source_path/04.jsonl,$source_path/05.jsonl,$source_path/06.jsonl,$source_path/07.jsonl,$source_path/08.jsonl, \
                 $source_path/09.jsonl,$source_path/10.jsonl,$source_path/11.jsonl,$source_path/12.jsonl,$source_path/13.jsonl,$source_path/14.jsonl,$source_path/15.jsonl,$source_path/16.jsonl,$source_path/17.jsonl, \
                 $source_path/18.jsonl,$source_path/19.jsonl,$source_path/20.jsonl,$source_path/21.jsonl,$source_path/22.jsonl,$source_path/23.jsonl,$source_path/24.jsonl,$source_path/25.jsonl,$source_path/26.jsonl, \
                 $source_path/27.jsonl,$source_path/28.jsonl,$source_path/29.jsonl"
   预处理脚本
   python ./tools/preprocess_data.py \
                --input ${source_data} \
                --output-prefix ${out_path} \
                --vocab ${word_path}/gpt2-vocab.json \ 
                --merge-file ${word_path}/gpt2-merges.txt \
                --dataset-impl mmap \
                --tokenizer-type GPT2BPETokenizer \
                --append-eod \
                --workers 150 \
                >  ./../logs/preprocess_pile_data.log  2>&1 &
   OR
   python ./tools/preprocess_data.py \
                --input ${source_data} \
                --output-prefix ${out_path} \
                --vocab ${word_path}/20B_tokenizer.json \
                --dataset-impl mmap \
                --tokenizer-type HFTokenizer \
                --append-eod \
                --workers 150 \
                >  ./../logs/preprocess_pile_data.log  2>&1 &
   备注：
   1、预计处理时间：36h
   2、官方训练使用：HFTokenizer

   ```
4. Finetuning
   ```
   GPT-NeoX加载.pt格式进行微调
   1、Finetuning时，在yml配置文件中添加配置添加
   "finetune":"True"
   参考：https://github.com/EleutherAI/gpt-neox/blob/v2.0/configs/neox_arguments.md
   
   2、微调加载ckpt:No optimizer states, for inference or finetuning, 39GB：
   https://the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/
   wget --cut-dirs=5 -nH -r --no-parent --reject "index.html*" https://the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/ -P 20B_checkpoints
   
   3、微调加载ckpt:Including optimizer states, 268GB
   https://the-eye.eu/public/AI/models/GPT-NeoX-20B/full_weights/
   wget --cut-dirs=5 -nH -r --no-parent --reject "index.html*" https://the-eye.eu/public/AI/models/GPT-NeoX-20B/full_weights/ -P 20B_checkpoints
   
   4、Hugging Face可加载格式的转换：
   python ./tools/convert_to_hf.py --input_dir /path/to/model/global_stepXXX --config_file your_config.yml --output_dir hf_model/save/location
    
    
   ```

# 适配代码

## Deepspeed代码

1. deepspeed/runtime/pipe/engine.py:957  #修正deepspeed 0.6.0配置 

   ```
   #inputs_grad_tail = [
                #    elt.grad for elt in inputs[1:] if elt.grad is not None
                #]   
                inputs_grad_tail = [elt.grad for elt in inputs[1:]]  # 修改后
   ```
   
## 模型代码修复

1. megatron/training.py:621 #deepspeed配置

   ```
   model=model,
            optimizer=optimizer,
            args=neox_args,
            lr_scheduler=_lr_scheduler,
            dist_init_required=False,
            model_parameters=_model_params,
            #config_params=neox_args.deepspeed_config, # 0.9.2 需要注释掉
            config_params=neox_args.deepspeed_config, # 0.6.0 不需要注释
            mpu=mpu if not neox_args.is_pipe_parallel else None,

   ```

2. megatron/training.py:628 

   ```
   if neox_args.is_pipe_parallel:
            model.set_has_attention_mask(True) # 0.6.0 不需要注释
            # model.set_has_attention_mask(True) # 0.9.2 需要注释掉
   ```

3. logits返回类型:megatron/training.py:346

   ```
   """Forward step."""
    if neox_args.is_pipe_parallel:
        # return model.eval_batch(data_iterator, return_logits=return_logits) # 适配0.9.2
        return model.eval_batch(data_iterator)  # 适配 0.6.0

   ```
4. hostfile适配:slots改为slot

   ```
    #127.0.0.1 slots=16 # 适配0.9.2
    127.0.0.1 slot=16 # 适配0.6.0
   ```
## 模型代码修改记录

1. 优化配置默认开启，在yaml文件中隐藏  megatron/neox_arguments/neox_args.py

   ```
    scaled_masked_softmax_fusion: bool = True # 默认 fused_softmax融合算子开启
    
    async_tensor_model_parallel_allreduce: bool = True # 默认allreduce通信隐藏开启

    use_triangle_attn: bool = True # 默认倒三角开启

   ```


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练

     ```
     python ./deepy.py train.py -d configs 20B.yml #修改20B.yml文件,默认0卡: "num_gpus": 1, "global_num_gpus": 1,  
     ```

   - 单机8卡训练

     启动8卡训练

     ```
     python ./deepy.py train.py -d configs 20B.yml # 适当调整 20B.yml文件，如20B单机8卡会oom，可将num-layer调小至22
     ```



3. 模型训练脚本参数说明如下。

   ```
   # Tokenizer /  checkpoint settings - you will need to change these to the location you have them saved in
    "vocab-file": "./tokenizer/20B_tokenizer.json", # 根据tokenizer_type 配置相应所需词表
    "save": "./20B_checkpoints", # ckpt保存路径
    "load": "./20B_checkpoints", # ckpt加载路径
    "data-path": "./data/pile_20B_tokenizer/pile_20B_tokenizer_text_document", # 数据集路径
    "pipe-parallel-size": 4, # 流水线并行
    "model-parallel-size": 2, # 模型并行，数据并行，自动计算
    # model settings
    "num-layers": 44, 
    "hidden-size": 6144,
    "num-attention-heads": 64,
    "seq-length": 2048,
    "max-position-embeddings": 2048,
    "norm": "layernorm",
    "pos-emb": "rotary",
    "rotary_pct": 0.25,
    # for all zero_optimization options, see https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training
    "zero_optimization": {
    "stage": 1,
    "allgather_partitions": True,
    "allgather_bucket_size": 1260000000,
    "overlap_comm": True,
    "reduce_scatter": True,
    "reduce_bucket_size": 1260000000,
    "contiguous_gradients": True,
    },
    # batch / data settings (assuming 96 GPUs)
    "train_micro_batch_size_per_gpu": 4, # per_batch_siize
    "gradient_accumulation_steps": 32, # 梯度累积
    # activation checkpointing
    "checkpoint-activations": true, # 重计算开关
    # misc. training settings
    "train-iters": 150000, # 训练step数据
    "checkpoint-factor": 500, # this variable previously called `save-interval` # ckpt保存间隔
    "eval-interval": 1000, # 1000步一预估
    "eval-iters": 10, #训练结束 预估
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 3**  训练结果展示表



|     NAME      | tflops | Iterations  | DataType  | Torch_Version | Card |
|:-------------:|:-------------:|:-:|:-:|:-:|:----:|
| GPU-2pp4mp2dp |     100      | 5000   | fp16  | 1.5  | A100 |
| NPU-2pp4mp2dp |     150      | 5000   | fp16  | 1.5  | 910 |


# 版本说明

## 变更

2023.07.07：首次发布。

## 已知问题

**_当前发行版本中存在的问题描述。_**
1. deepspeed问题pr:data helpers overflow bug https://github.com/NVIDIA/Megatron-LM/pull/507
相应适配:deepspeed/runtime/utils.py 636行
assert meta.dtype == torch.long ->  assert meta.dtype == torch.long or meta.dtype == torch.int32
2. 对shell脚本进行转码：find ./ -name '*' | xargs dos2unix

无。




