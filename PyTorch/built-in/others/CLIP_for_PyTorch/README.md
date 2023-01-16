# CLIP for PyTorch

-   [概述](#1)
-   [准备训练环境](#2)
-   [开始训练](#3)
-   [训练结果展示](#4)
-   [版本说明](#5)

# 概述

## 简述

[CLIP](https://openai.com/blog/clip/)(Contrastive Language-Image Pre-Training，以下简称 CLIP) 模型是 OpenAI 在 2021 年初发布的用于匹配图像和文本的预训练神经网络模型，是近年来在多模态研究领域的经典之作，可用于自然语言图像检索和zero-shot图像分类。

本文将介绍如何在COCO 2017数据集上进行CLIP模型的训练。

+ 参考实现：

  ```
  url=https://github.com/huggingface/transformers
  commit_id=d1d3ac94033b6ea1702b203dcd74beab68d42d83
  ```

+ 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/others
  ```

+ 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

+ 通过单击“立即下载”，下载源码包。

  

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                                           |
  |------------------------------------------------------------------------------| ------------------------------------------------------------ |
  | 硬件    | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)  |
  | 固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)  |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)                       |


- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```bash
  pip3 install -r requirements.txt
  ```

+ 安装transformers。

  ```bash
  cd transformers
  pip3 install -e ./
  cd ..
  ```

  

## 准备数据集

本文以COCO 2017数据集为例进行训练，用户可以通过以下命令下载COCO 2017数据集：

```bash
mkdir /opt/npu/dataset/coco
cd /opt/npu/dataset/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
```

数据集目录结构参考如下所示：

```
├── coco
      ├──train2017.zip
      ├──val2017.zip  
      ├──test2017.zip  
      ├──annotations_trainval2017.zip  
      └─ image_info_test2017.zip  
```

> **说明：** 
>
> 该数据集的训练脚本只作为一种参考示例，在使用其他数据集时，需要修改数据集路径。



## 获取预训练模型

本文使用[clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)预训练模型，执行以下命令获取预训练模型：

```bash
cd CLIP_for_PyTorch
python3.7 save_clip_roberta.py
```

执行以上代码，将会在CLIP_for_PyTorch目录生成clip-roberta文件夹，目录结构参考如下所示：

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

   ```bash
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练、单机8卡训练和多机训练。

   + 单机单卡训练

     启动单卡训练：

     ```bash
     bash test/train_clip_full_1p.sh --data_path=/opt/npu/dataset/coco --model_path=./clip-roberta --train_epochs=3    # 1p精度
     
     bash test/train_clip_performance_1p.sh --data_path=/opt/npu/dataset/coco --model_path=./clip-roberta --train_epochs=1    # 1p性能
     ```
   
   + 单机8卡训练
   
     启动8卡训练：
   
     ```bash
     bash test/train_clip_full_8p.sh --data_path=/opt/npu/dataset/coco --model_path=./clip-roberta --train_epochs=3    # 8卡精度
     
     bash test/train_clip_performance_8p.sh --data_path=/opt/npu/dataset/coco --model_path=./clip-roberta --train_epochs=1    # 8卡性能
     ```
     
   + 多机训练
     
     请参考[PyTorch模型多机多卡训练适配指南](https://gitee.com/ascend/pytorch/blob/master/docs/zh/PyTorch%E6%A8%A1%E5%9E%8B%E5%A4%9A%E6%9C%BA%E5%A4%9A%E5%8D%A1%E8%AE%AD%E7%BB%83%E9%80%82%E9%85%8D%E6%8C%87%E5%8D%97.md)中的“多机多卡训练流程”-“准备环境”章节进行环境设置，然后在每台服务器上使用如下命令启动训练：
     
     ```bash
     bash test/train_clip_cluster.sh --data_path=/opt/npu/dataset/coco --model_path=./clip-roberta --train_epochs=3 --nnodes=${机器总数量} --node_rank=${当前机器rank(0,1,2..)} --master_addr=${主服务器地址} --master_port=${主服务器端口号}
     ```
     
   + 训练脚本参数说明：
     
     ```bash
     --data_path:    coco数据集路径,和准备数据集章节中的coco文件夹路径保持一致
     --model_path:   预训练模型文件夹路径，和获取预训练模型章节生成的clip-roberta文件夹路径保持一致
     --train_epochs: 训练的epoch数
     --batch_size:   train和eval的batch size大小
     --model_name:   模型名称，默认为CLIP
     ```
     
   + 脚本中调用的python命令参数说明如下：
     
      ```bash
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

| NAME     | eval loss |   FPS   | AMP_Type | Epochs | Batch Size |
| -------- | :-------: | :-----: | :------: | ------ | ---------- |
| 1p-NPU   |  2.1984   | 25.486  |    O2    | 3      | 256        |
| 1p-竞品A |     -     |    -    |    O2    | 3      | 256        |
| 8p-NPU   |  1.5591   | 193.268 |    O2    | 3      | 64         |
| 8p-竞品A |  1.5565   | 177.471 |    O2    | 3      | 64         |
| 8p-NPU   | 2.2393   | 247.558 |    O2    | 3      | 256        |
| 8p-竞品A |     -     |    -    |    O2    | 3      | 256        |

# 版本说明

## 变更
2023.01.16：添加集群训练脚本说明

2023.01.12：Readme整改。

2022.12.20：首次发布。

## 已知问题

无。
