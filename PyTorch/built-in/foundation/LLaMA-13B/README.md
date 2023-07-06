# LLaMA-7B/13B for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

LLaMA是由Meta AI发布的大语言系列模型，完整的名字是Large Language Model Meta 
AI。LLaMA按照参数量的大小分为四个型号：LLaMA-7B、LLaMA-13B、LLaMA-30B与LLaMA-65B。LLaMA
模型的效果极好，LLaMA-13B在大多数基准测试中的表现都优于GPT-3（175B
），且无需使用专门的数据集，只使用公开可用的数据集即可至训练至最优。本工程基于FastChat仓，主要聚焦于LLaMA-7B/13B模型。

- 参考实现：

  ```
  url=https://github.com/lm-sys/FastChat.git
  commit_id=76f0424d1add61aadc8e5bdeed5ebe540f266ba3
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/foundation
  ```

# 准备训练环境

## 准备环境

默认配置需要每张卡有60G以上空闲内存。
- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------: |
  | PyTorch 1.11 | deepspeed 0.6.0 |

- 环境准备指导

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖

  在模型源码包根目录下执行以下命令，安装依赖。

  ```
  pip3 install --upgrade pip
  pip3 install einops sympy regex decorator scipy setuptools scm prompt toolkit
  ```
- 编译安装fschat

  在模型源码包根目录下执行命令，安装fachat库。
  
  ```
  pip3 install -e.
  ```

- 安装deepspeed及对应deepspeed_npu插件。
  
  在模型源码包根目录下执行以下命令，安装deepspeed。
  
  ```
  pip3 install deepspeed==0.6.0 
  git clone https://gitee.com/ascend/DeepSpeed.git
  cd DeepSpeed
  python setup.py develop
  ```
  使用whereis命令查看deepspeed安装路径/path/to/deepspeed/bin/deepspeed
  ，并将deepspeed_npu包导入。  

  打开"/path/to/deepspeed/bin/deepspeed"文件。
  
  ```
  vim /path/to/deepspeed/bin/deepspeed
  ```
  按“i”进入编辑模式，在“/path/to/deepspeed/bin/deepspeed”中增加以下内容。
  ```
  import deepspeed_npu
  
  ```
  按“ESC”键，输入:wq!，按“Enter”保存并退出编辑。

- 替换transformers库中相关文件
  
  将源码包根目录下transformers_modify文件夹中的各个文件分别替换到transformers
  安装目录下的对应位置（基于transformers 4.28.1版本）：
  ```
  training_args.py -> transformers/training_args.pu
  trainer.py -> transformers/trainer.py
  versions.py -> utils/versions.py
  modeling_llama.py -> transformers/models/llama/modeling_llama.py
  ```

- 安装pdsh（多机训练需要）

  deepspeed的多机训练需要安装pdsh，下载链接：https://github.com/chaos/pdsh/releases/download/pdsh-2.34/pdsh-2.34.tar.gz.

  安装方法如下：
  ```
  chmod 777 configure
  ./configure --with-ssh --build=arm-linux
  make
  make install
  ```
## 准备数据集

1. 获取数据集

   该任务以基于gpt3问答的数据集进行finetuning训练。

   以alpaca-data-conversation数据集为例，数据集结构参考如下所示。

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

2. 数据预处理
   
   基于上述格式的数据集无需预处理即可训练，若为其他对话数据集，则需修改为上述格式。

## 获取预训练模型

请参考原始仓库上的[README.md](https://github.com/lm-sys/FastChat/blob/76f0424d1add61aadc8e5bdeed5ebe540f266ba3/README.md)进行预训练模型获取。将获取的7B-vicuna/13B-vicuna预训练模型放至源码包根目录下。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机8卡训练和双机16卡训练。
   - 将数据集置于源码包根目录下playground/data文件夹内（若路径不存在请用户自行创建）。

   - 单机八卡训练（LLaMA-7B）

     ```
     bash ./7B_finetune.sh    
     ```

   - 双机16卡训练（LLaMA-13B）

     ```
     bash ./13B_finetune.sh  
     ```

   模型训练脚本参数说明如下。

   ```
    --model_name_or_path                       // 预训练参数路径 
    --data_path                                // 数据集路径 
    --fp16                                     // 参数使用fp16保存
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
    --tf32 False                               // 使用tf32训练，npu暂不支持  
    --model_max_length                         // 模型训练的sequence length
    --gradient_checkpointing                   // 是否开启重计算 
    --deepspeed                                // deepspeed配置脚本路径
   ```
   
   deepspeed参数说明如下。

   ```
    --fp16                                     // 混合精度训练相关配置 
    --optimizer                                // 优化器相关配置
    --zero_optimization                        // zero优化器相关配置
    --gradient_accumulation_steps              // 梯度累积步数
    --gradient_clipping                        // 梯度裁剪
    --train_batch_size                         // 训练batch size
    --train_micro_batch_size_per_gpu           // 训练micro batch size

   ```
   
   训练完成后，权重文件保存在output_dir下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 | FPS(tokens/s/p) | Epochs | Zero_Type |
|---------|-------|----------------:|--------|----------:|
| 7B-竞品A   | -     |            2452 | 3      |     zero1 |
| 7B-NPU  | -     |            2990 | 3      |     zero1 |
| 13B-竞品A  | -     |            1386 | 3      |     zero2 |
| 13B-NPU | -     |            1498 | 3      |     zero2 |

# 版本说明

## 变更

2023.07.05 首次发布。

## FAQ

无。