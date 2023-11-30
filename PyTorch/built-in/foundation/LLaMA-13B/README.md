# LLaMA-7B/13B for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述
## 简述

LLaMA是由Meta AI发布的大语言系列模型，完整的名字是Large Language Model Meta AI。LLaMA按照参数量的大小具有不同的型号。LLaMA模型的效果极好，无需使用专门的数据集，只使用公开可用的数据集即可至训练至最优。本工程基于FastChat仓，主要聚焦于LLaMA-7B/13B/33B模型。

- 参考实现：

  ```
  url=https://github.com/lm-sys/FastChat/tree/v0.2.31
  commit_id=9db21434b30a5355eb4723acc6562709f5ccc2c1
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/foundation
  ```
# 准备训练环境
- 环境准备指导

 - 请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

     **表 1**  环境配置表

    |  Software | Version  |
    | ------------ | ------------ |
    |  Pytorch |  2.1.0 |

- 这里要替换transformers库中的部分文件，使用下面命令时注意修改安装环境的路径。
   ```bash
    conda create -n test python==3.8
    conda activate test

    pip install torch==2.1.0
    pip install torch_npu-2.1.0xxxxx

    pip3 install --upgrade pip  # enable PEP 660 support
    pip3 install -e ".[model_worker,webui]"

    cp transformers_modify/modeling_llama.py /home/miniconda3/envs/test/lib/python3.8/site-packages/transformers/models/llama
    cp transformers_modify/training_args.py /home/miniconda3/envs/test/lib/python3.8/site-packages/transformers/
    cp transformers_modify/trainer.py /home/miniconda3/envs/test/lib/python3.8/site-packages/transformers/
    cp accelerate_modify/accelerator.py /home/miniconda3/envs/test/lib/python3.8/site-packages/accelerate
    cp accelerate_modify/dataclasses.py /home/miniconda3/envs/test/lib/python3.8/site-packages/accelerate/utils/
  ```
## 准备数据集
该任务以基于问答形式的数据集进行finetuning训练。 以alpaca数据集为例，数据集结构参考如下所示。注意保存数据集的路径，训练时修改脚本中的数据集路径。

   ```
   [
      {
        "id": "1",
        "conversations": [
          {
            "from": "human",
            "value": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGive three tips for staying healthy.\n\n### Response:"
          },
          {
            "from": "gpt",
            "value": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
          }
        ]
      },
      {
        "id": "2",
        "conversations": [
          {
            "from": "human",
            "value": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat are the three primary colors?\n\n### Response:"
          },
          {
            "from": "gpt",
            "value": "The three primary colors are red, blue, and yellow."
          }
        ]
      },
      ...
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。
## 获取预训练模型
这里可以参考原始仓库上的readme.md通过权重转换获取预训练模型，也可以从huggingface上获取预训练模型。注意保存预训练模型的位置，训练时修改脚本中的预训练模型路径。

参考预训练模型（huggingface上获取）：
   llama-7b:lmsys/vicuna-7b-v1.5；llama-13b:lmsys/vicuna-13b-v1.5。
# 开始训练
1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   模型支持单机8卡训练和双机16卡训练，重复训练时要删除之前训练保存的权重。

   - LLaMA-7B训练（单机8卡）

     ```
     bash ./scripts/train_vicuna_7b.sh    
     ```

   - LLaMA-13B训练（单机8卡）

     ```
     bash ./scripts/train_vicuna_13b.sh
     ```
   - LLaMA-13B双机训练要修改scripts/train_vicuna_13b.sh脚本

     ```
     torchrun --nproc_per_node=8 --master_port=20001 --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=90.90.3.79 fastchat/train/train_mem.py
     ```
     --nnodes：节点数；
     --node_rank：节点顺序(如0，1)；
     --master_addr：主节点ip。
   - 训练前注意修改环境变量路径，source环境信息
     ```
     source set_env.sh
     ```
    模型训练脚本参数说明如下，训练前注意修改预训练参数路径、数据集路径。

   ```
    --model_name_or_path                       // 预训练参数路径 
    --data_path                                // 数据集路径 
    --bf16                                     // 参数使用bf16保存
    --num_train_epochs                         // 训练epoch数
    --per_device_train_batch_size              // 每张卡上的训练batch size
    --per_device_eval_batch_size               // 每张卡上的评估batch size
    --gradient_accumulation_steps              // 梯度累积的步数
    --evaluation_strategy                      // 评估策略
    --save_strategy                            // ckpt保存策略
    --save_steps                               // ckpt保存间隔步数
    --save_total_limit                         // ckpt最大保存数量
    --learning_rate                            // 学习率
    --weight_decay                             // weight decay策略 
    --warmup_ratio                             // warmup步数的比例
    --lr_scheduler_type                        // 学习率衰减方式
    --logging_steps                            // 训练日志打印间隔步数
    --tf32 True                                // 使用tf32训练
    --model_max_length                         // 模型训练的sequence length
    --gradient_checkpointing                   // 是否开启重计算 
3. llama-33b 多机启动脚本配置（使用vicuna权重）

   多机微调启动脚本：`scripts/train_vicuna_33b_nnodes.sh`
   
   其中，部分配置参数需要根据实际情况进行配置：
    - `--nnodes`：要使用的机器数量；
    - `--nproc_per_node`：每台机器使用的NPU设备数量；
    - `--master_addr`：需要替换为主节点机器的IP地址；
    - `--node_rank`：主节点需要设置为0，其他节点`--node_rank`按顺序设置不重复即可；
    - `--master_port`：主节点的服务监听端口，可根据需要自行设置。

   启动时，各个节点都要执行`train_vicuna_33b_nnodes.sh`脚本来拉起微调任务。

# 训练结果展示
**表 2** 训练结果展示表


|  NAME | FPS(tokens/s/p)  | Epochs  |
| ------------ | ------------ | ------------ |
|  7B-NPU |  3120 | 3  |
|   7B-竞品A| 3120  |  3 |
| 13B-NPU(单机20层) | 1730 | 3  |
|  13B-竞品A(单机20层) | 1896  |  3 |

# 版本说明

## 变更

2023.11.23 首次发布。

## FAQ

无。
