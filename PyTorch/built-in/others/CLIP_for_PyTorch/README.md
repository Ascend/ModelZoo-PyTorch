# CLIP for PyTorch

-   [概述](#1)
-   [准备训练环境](#2)
-   [开始训练](#3)
-   [训练结果展示](#4)
-   [版本说明](#5)

# 概述

## 简述

CLIP (Contrastive Language-Image Pre-Training，以下简称 CLIP) 模型是 OpenAI 在 2021 年初发布的用于匹配图像和文本的预训练神经网络模型，是近年来在多模态研究领域的经典之作，可用于自然语言图像检索和zero-shot图像分类。

本文将介绍如何在COCO 2017数据集上进行CLIP模型的训练。

- 参考实现：

  ```
  url=https://github.com/huggingface/transformers
  commit_id=d1d3ac94033b6ea1702b203dcd74beab68d42d83
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/others
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```bash
  python3 -m pip install -r requirements.txt
  ```

- 安装transformers。

  ```bash
  python3 -m pip install -e transformers
  ```


## 准备数据集

1. 获取数据集。

   请用户自行获取数据集，上传到服务器任意路径下并解压。本文以COCO2017数据集为例进行训练，数据集目录结构参考如下所示：

   ```
   ├── coco
         ├──train2017.zip
         ├──val2017.zip  
         ├──test2017.zip  
         ├──annotations_trainval2017.zip  
         └─ image_info_test2017.zip  
   ```
   
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。


## 获取预训练模型

1. 本文使用**clip-vit-base-patch32**预训练模型，用户可在源码包根目录执行以下命令获取预训练模型。

   ```bash
   cd CLIP_for_PyTorch
   python3.7 save_clip_roberta.py
   ```

2. 执行以上代码，将会在CLIP_for_PyTorch模型根目录下生成clip-roberta文件夹，目录结构参考如下所示。

   ```
   ├──CLIP_for_PyTorch
       ├── clip-roberta
           ├── config.json
           ├── merges.txt
           ├── preprocessor_config.json
           ├── pytorch_model.bin
           ├── special_tokens_map.json
           ├── tokenizer_config.json
           ├── tokenizer.json
           └── vocab.json
   ```


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练、单机8卡训练和多机训练。

   - 单机单卡训练

     启动单卡训练。

     ```bash
     bash test/train_clip_full_1p.sh --data_path=/data/xxx/coco --model_path=./clip-roberta  # 单卡精度
     
     bash test/train_clip_performance_1p.sh --data_path=/data/xxx/coco --model_path=./clip-roberta  # 单卡性能
     ```
   
   - 单机8卡训练
   
     启动8卡训练。
   
     ```bash
     bash test/train_clip_full_8p.sh --data_path=/data/xxx/coco --model_path=./clip-roberta  # 8卡精度
     
     bash test/train_clip_performance_8p.sh --data_path=/data/xxx/coco --model_path=./clip-roberta  # 8卡性能
     ```
     
   - 多机训练
     
     请参考[PyTorch模型多机多卡训练适配指南](https://gitee.com/ascend/pytorch/blob/v1.5.0-3.0.rc2/docs/zh/PyTorch%E6%A8%A1%E5%9E%8B%E5%A4%9A%E6%9C%BA%E5%A4%9A%E5%8D%A1%E8%AE%AD%E7%BB%83%E9%80%82%E9%85%8D%E6%8C%87%E5%8D%97.md)中的“多机多卡训练流程”-“准备环境”章节进行环境设置，然后在每台服务器上使用如下命令启动训练。
     
     ```bash
     bash test/train_clip_cluster.sh --data_path=/data/xxx/coco --model_path=./clip-roberta --nnodes=${机器总数量} --node_rank=${当前机器rank(0,1,2..)} --master_addr=${主服务器地址} --master_port=${主服务器端口号}
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录，可参考上述启动命令；
   
   --model_path参数填写预训练模型文件夹路径，可参考上述启动命令。
   
   模型训练脚本参数说明如下。
   
   ```bash
   公共参数：
   --output_dir                                   // 训练结果和checkpoint保存路径
   --num_train_epochs                             // 训练的epoch次数
   --model_name_or_path                           // 预训练模型文件夹路径
   --data_dir                                     // 数据集路径
   --dataset_name                                 // 数据集名称
   --dataset_config_name                          // 数据集配置名称
   --image_column                                 // 图片所在的列
   --caption_column                               // 图片标题所在的列
   --remove_unused_columns                        // 是否删除未使用的列
   --do_train                                     // 执行训练
   --do_eval                                      // 执行评估
   --fp16                                         // 使用混合精度
   --dataloader_drop_last                         // 丢弃最后一个不完整的batch
   --fp16_opt_level                               // 混合精度级别
   --loss_scale                                   // 混合精度的loss_scale
   --use_combine_grad                             // 使用combine_grad选项
   --per_device_train_batch_size                  // 训练时使用的batch_size
   --per_device_eval_batch_size                   // 评估时使用的batch_size
   --learning_rate                                // 学习率
   --warmup_steps                                 // warmup steps，用于调整学习率
   --weight_decay                                 // 权重衰减值
   --overwrite_output_dir                         // 覆盖输出目录
   --local_rank                                   // 使用的卡id
   ```
   
   训练完成后，权重文件保存在output_dir路径下，并输出模型训练精度和性能信息。
   

# 训练结果展示

**表 2**  训练结果展示表

|    NAME    | eval loss |   FPS    | AMP_Type | Epochs | Batch Size |
| :--------: |:---------:|:--------:| :------: | :----: | :--------: |
|  1p-竞品A  |  1.7202   |    510   |    O2    | 3      | 64         |
|  1p-NPU    |  1.6863   |   440    |    O2    | 3      | 64         |
|  1p-NPU_arm|  1.6863   |   310    |    O2    | 3      | 64         |
|  8p-竞品A  |  1.5994   |   3340   |    O2    | 3      | 64         |
|  8p-NPU    |  1.5812   |   2800   |    O2    | 3      | 64         |
|  8p-NPU_arm|  1.5812   |   2200   |    O2    | 3      | 64         |


# 版本说明

## 变更

2023.03.03：更新readme，重新发布。

2023.01.31：GPU基线使用DDP进行测试。

2023.01.16：添加集群训练脚本说明。

2022.12.20：首次发布。

## FAQ

无。