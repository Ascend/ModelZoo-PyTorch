# Aquila pretrain for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [推理](#推理)
-   [版本说明](#版本说明)


# 概述

## 简述

悟道·天鹰（Aquila） 语言大模型是首个具备中英双语知识、支持商用许可协议、国内数据合规需求的开源语言大模型。
模型来源：北京智源研究院“悟道”人工智能大模型项目
能力优势：中文对话场景下对比其他模型效果更优 https://flageval.baai.ac.cn/#/trending

- 参考实现：

  ```
  url=https://github.com/FlagAI-Open/FlagAI
  commit_id=0b487694a8fa2eafe744b30c12f2a7785e60533a
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/foundation/FlagAI
  ```

# 准备训练环境

## 准备环境

- 环境准备指导。

  Python 版本 >= 3.8，建议使用 python 3.8

  PyTorch 版本 >= 1.8.0，建议使用 PyTorch 1.11.0

  安装昇腾Megatron（流程参考 https://gitee.com/ascend/Megatron-LM）

  安装昇腾Deepspeed（流程参考 https://gitee.com/ascend/DeepSpeed）


- 安装依赖。

  在FlagAI根目录下执行命令，安装flagai库以及所需要的依赖。
  ```shell
  git clone https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/foundation/FlagAI
  pip install -e .                    
  ```
  
  源码编译安装torch 1.11.0 和 torch_npu v1.11.0 （流程参考 https://gitee.com/ascend/pytorch/tree/v1.11.0/）
  
  源码编译安装apex（流程参考 https://gitee.com/ascend/apex）


- 准备模型配置文件：
    下载模型相关文件，下载链接：https://model.baai.ac.cn/model-detail/100098
    在模型仓库路径FlagAI/examples/Aquila/Aquila-pretrain下新建目录checkpoints_in/aquila-7b/，并将上述文件统一放到该文件夹下。


- 设置训练参数：

  参数文件位置信息：
  FlagAI/examples/Aquila/Aquila-pretrain/Aquila-pretrain.yaml
  ```
  batch_size: 2
  gradient_accumulation_steps: 1
  epochs: 600
  lr: 3.0e-4
  warm_up: 0.01
  save_interval: 1000
  log_interval: 1
  bmt_loss_scale: 131072
  save_optim: True
  save_rng: True
  eps: 1.e-8
  bmt_pre_load: True
  ```
  FlagAI/examples/Aquila/Aquila-pretrain/deepspeed.json
  ```
   {
    "train_micro_batch_size_per_gpu": 64,
    "gradient_accumulation_steps": 1,
    "steps_per_print": 100,
    "gradient_clipping": 1.0,
    "zero_optimization": {
      "stage": 2,
      "contiguous_gradients": true,
      "overlap_comm": true,
      "reduce_scatter": true,
      "reduce_bucket_size": 10e7,
      "allgather_bucket_size": 10e7,
      "cpu_offload": false
    },
    "scheduler": {
      "type": "WarmupLR",
      "params": {
          "warmup_min_lr": 1e-6,
          "warmup_max_lr": 1e-6,
          "warmup_num_steps": 2000
      }
   },
    "zero_allow_untested_optimizer": true,
    "fp16": {
      "enabled": true,
      "loss_scale": 8192,
      "loss_scale_window": 1000,
      "hysteresis": 2,
      "min_loss_scale": 1
    },
    "optimizer": {
      "type": "Adam",
      "params": {
        "lr": 1e-5,
        "weight_decay": 0.1,
        "betas": [
          0.9,
          0.98
        ],
        "eps": 1e-6
      }
    },
    "activation_checkpointing": {
      "partition_activations": true,
      "contiguous_memory_optimization": false
    },
    "wall_clock_breakdown": false
   }
  ```
- FlagAI/examples/Aquila/Aquila-pretrain/ checkpoints_in/aquila-7b/config.json
  ```
  {
  "dim": 4096,
  "multiple_of": 256,
  "n_heads": 32,
  "n_layers": 32,
  "norm_eps": 1e-05,
  "vocab_size": 100008,
  "max_seq_len": 2048,
  "initializer_range": 0.006,
  "flash_atten": false,
  "flash_atten_llama_style": false,
  "checkpoint_activations": true,
  "ckpoint_layer": 26,
  "use_triangle_attn": true
  }
  ```

- 运行脚本修改:
   
  修改文件：FlagAI-master\examples\Aquila\Aquila-pretrain\bmtrain_mgpu.sh

  NCCL_SOCKET_IFNAME修改为当前机器的网卡端口
  
  NODE_ADDR修改为当前机器的网络ip

  新增hostfile文件：FlagAI-master\examples\Aquila\Aquila-pretrain\hostfile
里面添加一行信息如下：
  ```
  90.90.90.120(当前机器的网络ip) slots=8
   ```

- 修改三方库代码：

  注释文件path/Megatron-LM/megatron/utils.py line 11 的import amp_C

  修改文件path/Megatron-LM/megatron/data/indexed_dataset.py 中所有的np.float为np.float32或者np.float64



## 准备数据集
- 数据集准备：

  数据集 alpaca-data-conversation （https://github.com/lm-sys/FastChat/blob/v0.1.10/playground/data/alpaca-data-conversation.json）
  
  使用aquila自带的make_indexed_dataset.sh脚本将数据处理成.bin和.idx格式文件（为保证中英文训练，将alpaca-data-conversation数据和aquila官方demo数据合并测试）。处理完成的.bin和.idx文件数据需要放到../../indexed_dataset/data/目录下：


# 开始训练
- 拉起训练：

  在 flagAI/examples/Aquila/Aquila-pretrain/ 下运行
   ```shell
  bash local_trigger_docker.sh hostfile Aquila-pretrain.yaml aquila-7b aquila_experiment
   ```

# 训练结果展示

**表 2**  训练结果展示表

|               | batch_size | seq_len | 耗时/s | 8卡吞吐量 (tokens/s) |
|---------------|------------|---------|------|------------------|
| 8p-竞品A1       | 2          | 2048    | 2.30 | 14247            |
| 8p-竞品A8       | 2          | 2048    | 2.45 | 13375            |
| 8p-NPU-910B3 | 2          | 2048    | 2.35 | 13944            |


# 推理

  在 flagAI/examples/Aquila/Aquila-pretrain/ 下运行
   ```shell
   python generate.py
   ```
   即可体验交互式文本生成


输入:将进酒·君不见 唐·李白

![img.png](img.png)


# 版本说明

## 变更

2023.09.10：首次发布。



   
