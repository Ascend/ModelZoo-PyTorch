# FocalTransformer for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

FocalTransformer是一个图像分类网络，网络使用粗粒度和细粒度两种模式分别汇聚远距离和近距离的token信息，并且使用多尺度的金字塔结构，这使得它比ViT更有效地聚合全局信息，同时计算复杂度不会超出ViT太多。

- 参考实现：

  ```
  url=https://github.com/microsoft/Focal-Transformer.git
  commit_id=57bb3031582a2afb2d2a6916612bc4311316f9fc
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，将数据集上传到服务器任意路径下并解压。 数据集目录结构如下所示：

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

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   ```
   # training 8p results
   bash ./test/train_full_8p.sh --data_path=real_data_path
   
   # training 1p performance
   bash ./test/train_performance_1p.sh --data_path=real_data_path
   
   # training 8p performance
   bash ./test/train_performance_8p.sh --data_path=real_data_path
   
   # finetune 1p
   bash ./test/train_finetune_1p.sh --data_path=real_data_path --finetune_model=real_checkpoint_path
   ```

   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --cfg                               //config路径
   --data-path                          //数据集路径
   --batch-size                        //训练批次大小
   --stop_step
   finetune参数：
   --finetune_switch                   //finetune开关
   --finetune_model                    //用户训练模型的存放路径
   ```
   
   训练完成后，权重文件保存在output文件夹下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

由于存在算子性能问题，仅训练7个epoch判断精度是否对齐。

| NAME    | Acc@1          |    FPS | Epochs  | AMP_Type |
| ------- | -------------- | -----: | ------- | -------: |
| 1p-竞品 | -              |  94.77 | 1       |       O1 |
| 1p-NPU  | -              |   9.32 | 1       |       O1 |
| 8p-竞品 | 34.43% (83,6%) | 703.54 | 7 (300) |       O1 |
| 8p-NPU  | 34.24%         |  73.84 | 7       |       O1 |

# 版本说明

## 变更

2022.09.24：首次发布。

## 已知问题

无。