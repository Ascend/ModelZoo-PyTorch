# sd-scripts-xl for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

Stable Diffusion(SD)是计算机视觉领域的一个生成式大模型，能够进行文生图（txt2img）和图生图（img2img）等图像生成任务。sd-scripts仓适配了SD模型的训练、生成以及多个下游任务脚本，包括新版本**Stable Diffusion XL**。

- 参考实现：

  ```
  url=https://github.com/kohya-ss/sd-scripts
  commit_id=46cf41cc93d5856664a2835da2d92796f9344281
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/diffusion/
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.11+python3.8 | diffusers==0.21.2 accelerate==0.23.0 fairscale==0.4.13 torchvision==0.12.0 torchvision_npu==0.12.0 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》搭建torch环境。
  
- 安装依赖。

  1. 在模型根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  
     ```
     pip install -r requirements.txt 
     ```
  
  2. 参考[gitee官仓](https://gitee.com/ascend/vision/tree/v0.12.0-dev/)安装对应0.12.0版本的torchvision和torchvision_npu。
  
- 替换三方库补丁。

  在模型根目录下，将以下命令中`python_path`变量赋值为当前环境下的python路径，并执行：

  ```
  python_path=/path/lib/python3.8/site-packages
  \cp $python_path/accelerate/accelerator.py $python_path/accelerate/accelerator.py_bak
  \cp $python_path/accelerate/scheduler.py $python_path/accelerate/scheduler.py_bak
  \cp $python_path/fairscale/optim/oss.py $python_path/fairscale/optim/oss.py_bak
  
  \cp ./third_patch/accelerate_patch/accelerator.py $python_path/accelerate/accelerator.py
  \cp ./third_patch/accelerate_patch/scheduler.py $python_path/accelerate/scheduler.py
  \cp ./third_patch/fairscale_patch/oss.py $python_path/fairscale/optim/oss.py
  ```

  


## 准备数据集

1. 联网情况下，数据集会自动下载。

2. 无网络情况下，用户需自行获取laion数据集，并在shell启动脚本中将`dataset_name`参数，设置为本地数据集的绝对路径，填写一级目录。

   数据结构如下：

   ```
   $dataset
   ├── 000xx.tar
   	├── 000xxxxx.jpg
   	├── 000xxxxx.json
   	└── train-0001.txt
   ├── 000xx.parquet
   └── 000xx_stats.json
   ```
   
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。
   
   

## 获取预训练模型

1. 联网情况下，预训练模型会自动下载。

2. 无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

   ```
   CompVis/stable-diffusion-v1-4
   runwayml/stable-diffusion-v1-5
   stabilityai/stable-diffusion-2
   stabilityai/stable-diffusion-2-1
   ```

3. 获取对应的预训练模型后，在shell启动脚本中将`model_name`参数，设置为本地预训练模型路径，填写一级目录。



# 开始训练

## 预训练模型

本节以预训练为例，展示模型训练方法，其余下游任务txtimg、lora、controlnet、dreambooth、textual inversion等可自行参考适配预训练脚本。

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练、单机8卡和多机多卡预训练

   - 单机单卡预训练

     ```shell
     bash test/pretrain_full_1m_1p_sdxl.sh # 1卡精度，默认为混精，带FA场景
     bash test/pretrain_full_1m_1p_sdxl.sh --max_train_epoch=13 # 1卡性能，默认为混精，带FA场景
     ```
     
   - 单机8卡预训练
   
     ```shell
     bash test/pretrain_full_1m_8p_sdxl.sh # 8卡精度，默认为混精，带FA场景
     bash test/pretrain_full_1m_8p_sdxl.sh --max_train_epoch=13 # 8卡性能，默认为混精，带FA场景
     ```
     
   - 多机多卡预训练
   
     ```shell
     bash test/pretrain_full_nm_np_sdxl.sh --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_ip=x.x.x.x --master_port=8989 # 多卡精度，默认为混精，带FA场景
     bash test/pretrain_full_nm_np_sdxl.sh --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_ip=x.x.x.x --master_port=8989 --max_train_epoch=13 # 多卡性能，默认为混精，带FA场景
     ```
     
     > 脚本参数说明：
     >
     > nnodes：机器数量。
     >
     > nproc_per_node：每台机器的卡数。
     >
     > node_rank：当前机器是几号机器，主机为0，其它为1,2...
     >
     > master_ip：主机ip地址。
     >
     > master_port：主机端口号。
     
   - 模型的python训练脚本参数说明。
   
   ```shell
   sdxl_pretrain.py：
   --max_train_steps                   //最大训练步数
   --max_train_epoch=120               //最大训练轮数
   --pretrained_model_name_or_path     //预训练模型名称或者绝对路径
   --vae                               //vae模型名称或者绝对路径
   --tokenizer1_path                   //tokenizer1名称或者绝对路径
   --tokenizer2_path                   //tokenizer1名称或者绝对路径
   --train_data_dir                    //数据集名称或者绝对路径
   --resolution                        //图片分辨率
   --enable_bucket                     //使能数据集中图片分辨率分桶操作
   --min_bucket_reso                   //分桶操作的最小分辨率
   --max_bucket_reso                   //分桶操作的最大分辨率
   --output_dir                        //输出ckpt的输出路径
   --output_name                       //输出ckpt的前缀
   --save_every_n_epochs               //每n个epoch保存一次ckpt
   --save_precision                    //保存ckpt的精度
   --logging_dir                       //输出日志路径
   --gradient_checkpointing            //使能重计算
   --gradient_accumulation_steps       //梯度累计步数
   --learning_rate                     //学习率
   --train_text_encoder                //使能训练text_encoder
   --learning_rate_te1                 //text_encode1的学习率
   --learning_rate_te2                 //text_encode2的学习率
   --lr_warmup_steps                   //学习率预热步数
   --max_grad_norm                     //最大梯度归一值
   --lr_scheduler                      //学习率策略
   --lr_scheduler_num_cycles           //学习率策略中周期数量
   --train_batch_size                  //设置batch_size
   --mixed_precision                   //精度模式
   --seed                              //随机种子
   --caption_extension                 //caption文件的扩展名
   --shuffle_caption                   //打乱caption
   --keep_tokens                       //打乱caption token时保持前N个token不变（token用逗号分隔）
   --optimizer_type                    //优化器类型
   --max_token_length                  //最大token长度
   --enable_npu_flash_attention        //使能npu_flash_attention，仅支持fp16精度
   ```
   
   训练完成后，权重文件保存在`test/output`路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | sd版本 | FPS  | batch_size | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :------: | :-----------: | :-----------: |
| 8p-竞品A | xl | 20.2 | 4 | fp16 |      1.13      |
|  8p-NPU-910  | xl | 10.4 | 4 | fp16 |      1.11      |

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 版本说明

## 变更

2023.12.11：首次发布。
