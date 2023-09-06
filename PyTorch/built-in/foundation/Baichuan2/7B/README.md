# Baichuan2-7B for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

Baichuan2-7B 是由百川智能开发的一个开源可商用的大规模预训练语言模型。基于 Transformer 结构，在大约 1.2 万亿 tokens 上训练的 70 亿参数模型，支持中英双语，上下文窗口长度为 4096。在标准的中文和英文 benchmark（C-Eval/MMLU）上均取得同尺寸最好的效果。



# 准备训练环境

## 准备环境

默认配置需要每张卡有60G以上空闲内存。
- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Python版本 | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------: | :----------------------------------------------: |
  | Python 3.7 | PyTorch 1.11 | deepspeed 0.9.2 |

- 环境准备指导

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
  **注: Baichuan2基于BF16训练，确保获取的CANN支持BF16**
  
- 创建conda环境

  ```shell
  conda create -n py37 python=3.7
  conda activate py37
  ```

- 安装依赖

  ```shell
pip3 install --upgrade pip
  pip3 install einops sympy regex decorator scipy setuptools-scm prompt-toolkit attrs accelerate sentencepiece transformers==4.28.1
  ```
  
- 安装deepspeed及对应deepspeed_npu插件
  
  在模型源码包根目录下执行以下命令，安装deepspeed。
  
  ```shell
  pip3 install deepspeed==0.9.2 
  git clone https://gitee.com/ascend/DeepSpeed.git
  cd DeepSpeed
  python setup.py develop
  ```
  
- 安装昇腾torch及对应torch_npu插件
  
  ```shell
  pip3 install torch-1.11.0-cp37-cp37m-linux_aarch64.whl
  pip3 install torch_npu-1.11.0.post1-cp37-cp37m-linux_aarch64.whl
  ```


- 替换transformers库中相关文件
  
  将源码包根目录下transformers_modify文件夹中的各个文件分别替换到transformers库安装目录下的对应位置（基于**transformers 4.28.1**版本）：
  
  ```shell
  training_args.py -> transformers/training_args.py
  trainer.py -> transformers/trainer.py
  versions.py -> utils/versions.py
  ```


## 准备数据集

本仓库以外卖评论情感识别的任务为例，演示如何基于Baichuan2-7B模型完成全参微调

1. 获取数据集
    该任务采用开源中文外卖数据集[waimai_10k](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/waimai_10k/waimai_10k.csv)进行finetuning训练，原始数据结构参考如下所示，其中1表示积极，0表示消极。

   | **label** |                 **review**                  |
   | :-------: | :-----------------------------------------: |
   |     1     |         送餐特别快,态度也好,辛苦啦          |
|     0     |            难吃!!!油死了，味道烂            |
   |     0     | 今天菜太咸，连着定了3天吃，一天比一天难吃。 |
   |     0     |           送的太慢了，菜都凉了。            |

2. 数据预处理

   下载`waimai_10k.csv`后放在源码包根目录下，运行源码包根目录下的`make_data.py`脚本将原始数据处理成处理成json格式，提取前10000条作为训练集`train.jsonl`，剩余样本作为验证集`eval.jsonl`，脚本生成的样本格式如下：

   ```json
   {"review": "11:30,的餐，下午三点才送到.,还是打百度投诉的结婚！", "label": "消极"}
   {"review": "还不错,送货很快味道也还好", "label": "积极"}
   {"review": "等了五十分钟，受不了的龟速", "label": "消极"}
   {"review": "很快,很好", "label": "积极"}
   {"review": "超级好超级好超级好啊", "label": "积极"}
   ```


## 获取预训练模型

下载 [Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base) 的模型权重文件，用源码包根目录下的`Baichuan2-7B/modeling_baichuan.py`替换下载下来的模型权重文件夹中的`modeling_baichuan.py`文件。



# 开始训练

## 训练模型

1. 进入解压后的源码包根目录

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本

   该模型支持单机8卡训练

   注意：

   - 需要将`ds_run_bf16.sh`文件开头的CANN安装目录替换成实际运行环境下的CANN目录，同时确保CANN支持BF16特性
   - 需要将`model_name_or_path`指向从huggingface下载的模型权重文件目录
   - 确保数据集为与源码包根目录下的data文件夹内（若路径不存在请用户自行创建）

   ```
    bash ./ds_run_bf16.sh    
   ```

   模型训练脚本参数说明如下：

   ```
    --model_name_or_path                       // 预训练参数路径 
    --train_data                               // 训练数据集路径 
    --eval_data                                // 验证数据集路径
    --bf16                                     // 参数使用bf16保存
    --output_dir                               // ckpt保存位置
    --num_train_epochs                         // 训练epoch数
    --per_device_train_batch_size              // 每张卡上的训练batch size
    --gradient_accumulation_steps              // 梯度累积的步数
    --gradient_checkpointing                   // 是否开启重计算 
    --save_strategy                            // ckpt保存策略
    --learning_rate                            // 学习率
    --weight_decay                             // weight decay策略 
    --warmup_ratio                             // warmup步数的比例
    --lr_scheduler_type                        // 学习率衰减方式
    --logging_steps                            // 训练日志打印间隔步数
    --tf32 False                               // 使用tf32训练，npu暂不支持  
    --model_max_length                         // 模型训练的sequence length
    --deepspeed                                // deepspeed配置脚本路径
   ```

   deepspeed参数说明如下：

   ```
    --bf16                                     // bf16训练相关配置 
    --optimizer                                // 优化器相关配置
    --zero_optimization                        // zero优化器相关配置
    --gradient_accumulation_steps              // 梯度累积步数
    --gradient_clipping                        // 梯度裁剪
    --train_batch_size                         // 训练batch size
    --train_micro_batch_size_per_gpu           // 训练micro batch size
   ```

   训练完成后，权重文件默认保存在源码包根目录下的`outputs`目录下。

   

# 模型推理

1. 进入解压后的源码包根目录

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行测试脚本

   注：需要将`run_test.sh`文件开头的CANN安装目录替换成实际运行环境下的CANN目录，同时确保CANN支持BF16特性

   ```
   bash ./run_test.sh    
   ```

   如果微调成功，可以看到类似下表的结果：

   ​									     **模型对比：base对话模型 vs tune情感识别模型**

   | review                                             | label | base                   | tune |
   | -------------------------------------------------- | :---: | ---------------------- | ---- |
   | 11:30,的餐，下午三点才送到.,还是打百度投诉的结婚！ | 消极  | 抱歉，作为一个大语言   | 消极 |
   | 还不错,送货很快味道也还好                          | 积极  | 非常感谢您的满意评价   | 积极 |
   | 等了五十分钟，受不了的龟速                         | 消极  | 作为一个大语言模型     | 消极 |
   | 很快，很好                                         | 积极  | 非常好，谢谢！请问     | 积极 |
   | 超级好超级好超级好啊                               | 积极  | 非常好，非常好，非常好 | 积极 |

   

# 版本说明

## 变更

2023.09.06 首次发布

## FAQ

无