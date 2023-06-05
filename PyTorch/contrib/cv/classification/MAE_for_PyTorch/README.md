# MAE for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)



# 概述

## 简述

掩码自编码器(MAE)是一种自监督方法，通过从可见的RGB patches重建掩码的RGB patches来预训练ViT。
MAE的设计虽然简单，但已被证明是一个强大的、可扩展的视觉表示学习预训练框架。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/mae
  commit_id=efb2a8062c206524e35e47d04501ed4f544c0ae8
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.2.2.post3；pillow==8.4.0 |
  | PyTorch 1.8 | torchvision==0.9.1；pillow==9.1.0 |
  | PyTorch 1.11 | timm==0.4.5 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  
  pip install -r 1.11_requirements.txt  # PyTorch1.11版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。

## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，CIFAR-10等，将数据集上传到服务器任意路径下并解压。

   以ImageNet2012数据集为例，数据集目录结构参考如下所示。

   ```
   ├── ImageNet2012
         ├──train
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...   
              ├──...                     
         ├──val  
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...              
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。



# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。
   
   1. 预训练
    ```bash
    # pre-training 1p performance，单p上运行1个epoch，运行时间约为1h
    # 输出性能日志./output_pretrain_1p/910A_1p_pretrain.log、总结性日志./output_pretrain_1p/log.txt
    bash ./test/pretrain_performance_1p.sh --data_path=real_data_path
    
    # pre-training 8p performance，8p上运行1个epoch，运行时间约为9min
    # 输出性能日志./output_pretrain_8p/910A_8p_pretrain.log、总结性日志./output_pretrain_8p/log.txt
    bash ./test/pretrain_performance_8p.sh --data_path=real_data_path
    
    # pre-training 8p full，8p上运行400个epoch，运行时间约为60h
    # 输出完整预训练日志./output_pretrain_full_8p/910A_8p_pretrain_full.log、总结性日志./output_pretrain_full_8p/log.txt
    bash ./test/pretrain_full_8p.sh --data_path=real_data_path
    ```
   2. fine-tuning
   
    ```bash
    # fine-tuning 1p performance，单p上运行1个epoch，运行时间约为1h15min，
    # 输出性能日志./output_finetune_1p/910A_1p_finetune.log、总结性日志./output_finetune_1p/log.txt
    bash ./test/finetune_performance_1p.sh --data_path=real_data_path --finetune_pth=pretrained_model_path
    
    # fine-tuning 8p performance，8p上运行1个epoch，运行时间约为11min
    # 输出性能日志./output_finetune_8p/910A_8p_finetune.log、总结性日志./output_finetune_8p/log.txt
    bash ./test/finetune_performance_8p.sh --data_path=real_data_path --finetune_pth=pretrained_model_path
    
    # fine-tuning 8p full，8p上运行100个epoch，运行时间约为18h
    # 输出完整微调日志./output_finetune_full_8p/910A_8p_finetune_full.log、总结性日志./output_finetune_full_8p/log.txt
    bash ./test/finetune_full_8p.sh --data_path=real_data_path --finetune_pth=pretrained_model_path

    # fine-tuning_large 8p performance，8p上运行1个epoch，910B运行时间约为14min
    bash ./test/finetune_performance_large_8p.sh --data_path=real_data_path --finetune_pth=pretrained_model_path

    # fine-tuning_large 8p full，8p上运行50个epoch，910B运行时间约为12h
    bash ./test/finetune_full_large_8p.sh --data_path=real_data_path --finetune_pth=pretrained_model_path

    # fine-tuning_large 16p full，16p上运行50个epoch，910B运行时间约为6h
    bash ./test/finetune_full_large_16p.sh --data_path=real_data_path --finetune_pth=pretrained_model_path
    
    # 8p Base_eval，运行时间约为3min
    # 输出eval日志./output_finetune_eval_8p/910A_8p_finetune_eval.log
    bash ./test/finetune_eval_8p.sh --data_path=real_data_path --resume_pth=finetuned_model_path
    ```

    说明：MAE-Large配置在910B上进行训练，16p脚本需要在启动脚本修改节点IP

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                              // 数据集路径
   --finetune_pth                           // 预训练模型路径
   --resume_pth                             // finetuned模型路径
   ```


# 训练结果展示

**表 2**   MAE-Base Pre-Training Result

| NAME | LOSS | FPS | Epochs   | AMP_Type | Torch_Version |
| :------: | :------:  | :------: | :------: | :------: | :------: |
| 1p-竞品V  | -      | 320   | 1        | -       | 1.5    |
| 1p-NPU | -     | 328  | 1      | O2      | 1.8  |
| 8p-竞品V | 0.4107 | 2399 | 400 | - | 1.5 |
| 8p-NPU | 0.4107 | 2515 | 400 | O2 | 1.8 |

**表 3**   MAE-Base Fine-Tuning Result

| NAME | Acc@1 | FPS | Epochs   | AMP_Type | Torch_Version |
| :------: | :------:  | :------: | :------: | :------: | :------: |
| 1p-竞品V  | -      | 218   | 1        | -       | 1.5    |
| 1p-NPU | -     | 306   | 1      | O2      | 1.8   |
| 8p-竞品V | 83.07 | 1538 | 100 | - | 1.5 |
| 8p-NPU | 83.34 | 2263 | 100 | O2 | 1.8 |

**表 4**   MAE-Large Fine-Tuning Result

| NAME | Acc@1 | FPS | Epochs   | AMP_Type | Torch_Version |
| :------: | :------:  | :------: | :------: | :------: | :------: |
| 8p-竞品A | 85.85 | 1190 | 50 | - | 1.8 |
| 8p-NPU | 83.86 | 1603 | 50 | O2 | 1.8 |
| 16p-NPU | 85.97 | 3145 | 50 | O2 | 1.8 |

说明：MAE-Large配置在910B上进行训练

# 版本说明

## 变更
2023.06.05：添加MAE-Large Fine-Tuning配置

2022.12.22：更新readme，重新发布。

## FAQ

无。

