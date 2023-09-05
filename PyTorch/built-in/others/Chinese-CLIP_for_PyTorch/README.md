# ChineseCLIP for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

本项目为CLIP模型的中文版本，使用大规模中文数据进行训练（~2亿图文对），旨在帮助用户快速实现中文领域的图文特征&相似度计算、跨模态检索、零样本图片分类等任务。
- 参考实现：

  ```
  url=https://github.com/OFA-Sys/Chinese-CLIP
  commit_id=2c38d03557e50eadc72972b272cebf840dbc34ea
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

  | Torch_Version   | 三方库依赖版本   |
  | :--------: | :-----: |
  | PyTorch 1.11 | - |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  ```
  pip install -r requirements.txt
  python3 -m pip install -e .
  ```

## 准备预训练模型与词表

- 参考实现中提供了预训练模型的下载链接：
- VIT-B-16：https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-b-16.pt
- VIT-H-14：https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-h-14.pt
- 下载好的预训练模型放在pretrained_weights路径下

## 准备数据集

- 参考实现中提供了处理好的Flickr30K-CN数据集的下载链接：
- Flickr30K-CN：https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/Flickr30k-CN.zip
- 下载好的预训练模型放在datasets路径下



# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash test/flickr30k_finetune_vit-b-16_rbt-base_full_1p.sh .        # 单卡训练 vit-b
     bash test/flickr30k_finetune_vit-h-14_rbt-large_full_1p.sh .       # 单卡训练 vit-h
     
     bash test/flickr30k_finetune_vit-b-16_rbt-base_performance_1p.sh .        # 单卡性能 vit-b
     bash test/flickr30k_finetune_vit-h-14_rbt-large_performance_1p.sh .       # 单卡性能 vit-h
     ```
     
   - 单机8卡训练

     启动8卡训练。
     ```
     bash test/flickr30k_finetune_vit-b-16_rbt-base_full_1p.sh .        # 8卡训练 vit-b
     bash test/flickr30k_finetune_vit-h-14_rbt-large_full_1p.sh .       # 8卡训练 vit-h
     
     bash test/flickr30k_finetune_vit-b-16_rbt-base_performance_1p.sh .        # 8卡性能 vit-b
     bash test/flickr30k_finetune_vit-h-14_rbt-large_performance_1p.sh .       # 8卡性能 vit-h
     ```
     
  
   
   训练完成后，权重文件保存在./experiments下，并输出模型训练精度和性能信息。


# 训练结果展示

  **表 2**  训练结果展示表


  | pretrain_model |  NAME  | image_to_text_R@5 | text_to_image_R@5 |   FPS   | Epochs | batch_size |
|:--------------:|:------:|:-----------------:|:-----------------:|:-------:|:------:|:----------:|
  |     vit-b      | 8p-NPU |       94.76       |       98.7        | 2280.00 |   3    |    128     |
  |     vit-b      | 8p-竞品V |       94.63       |       98.97       | 2512.40 |   3    |    128     |
  |     vit-h      | 8p-NPU |       95.18       |       98.7        | 316.07  |   3    |     32     |
  |     vit-h      | 8p-竞品V |       95.34       |       99.4        | 348.13  |   3    |     32     |


# 版本说明

## 变更

2023.08.29：首次提交。
2023.09.04：适配NPU，新增训练和性能脚本。

## FAQ

无。