# ICNet_ID1781 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

ICNet是一个基于PSPNet的实时语义分割网络，设计目的是减少PSPNet推断时期的耗时，在PSPNet的基础上引入级联特征融合模块，实现快速且高质量的分割模型。在Cityscapes数据集上进行了相关实验。

- 参考实现：

  ```
  url=https://github.com/hszhao/ICNet.git
  commit_id=dd2bf78ff3f9ff05f6d35c40a3ee216158d9e892
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/semantic_segmentation
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
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括Cityscapes等，将数据集上传到服务器任意路径下并解压。

   以Cityscapes数据集为例，数据集目录结构参考如下所示。

   ```
   ├── cityscapes
         ├── gtFine
              ├── train
              ├── val
              └── test
         ├── leftImg8bit
              ├── train
              ├── val
              └── test
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

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=/data/xxx/
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/
     ```

   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --cityscapes_root                   //数据集路径
   --name                              //使用模型，默认：icnet
   --backbone                          //使用模型主干，默认：resnet50
   --base_size                         //图像数据处理最小边长，默认：1024
   --crop_size                         //图像数据处理裁剪边长，默认：960
   --epochs                            //重复训练次数，默认：200
   --train_batch_size                  //训练批次大小，默认：16
   --valid_batch_size                  //测试批次大小，默认：16
   --init_lr                           //初始学习率，默认：0.01
   --momentum                          //动量，默认：0.9
   --weight_decay                      //权重衰减，默认：0.0001
   多卡训练参数：
   --is_distributed                    //是否使用多卡训练
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | mIoU |  FPS | Epochs | AMP_Type |
| ------- | ----- | ---: | ------ | -------: |
| 官方源码      | 68.4    | -    | 200    | -        |
| 1p-竞品(GPU) | -      | 9.3  | -   | -        |
| 1p-NPU-ms1.5  | -     | 6.0 | -   | -       |
| 1p-NPU-ms1.8  | -     | 10.1 | - | -        |
| 8p-竞品(GPU) | 68.3  | 83     | 200    | -        |
| 8p-NPU-ms1.5  | 68.1  | 81  | 200    | O1       |
| 8p-NPU-ms1.8  | 68.9  | 147.2 | 200    | O1       |

# 版本说明

## 变更

2022.08.14：更新pytorch1.8版本，重新发布。

2020.03.08：首次发布。

## 已知问题

一、精度无法达到官网指标

- 说明
    - 官网中，模型指标标注的是mIoU 71%，但是实测结果只有68.4%，无法达到官网指标。
      官网参考Base version of the model from [the paper author's code on Github](https://github.com/liminn/ICNet-pytorch).
    - GPU-8p精度对齐过程，通过调整超参，最终精度和最优精度分别如下：

    | batch_size   | lr       | loss_scale | opt_level  | final mIoU  | best mIoU   |
    | :---------:  | :------: | :------:   | :--------: | :---------: | :---------: |
    | 16           | 0.06     | 128        |  O1        | 68.1        |  68.7       |
    | 16           | 0.08     | 128        |  O1        | 68.3        |  68.6       |
    | 16           | 0.1      | 128        |  O1        | 67.7        |  68.6       |
    | 16           | 0.08     | 128        |  O2        | 67.7        |  68.0       |

    - NPU-8p精度对齐过程，通过调整超参，最终精度和最优精度分别如下：

    | batch_size   | lr       | loss_scale | opt_level  | final mIoU  | best mIoU   |
    | :---------:  | :------: | :------:   | :--------: | :---------: | :---------: |
    | 16           | 0.08     | 32         |  O1        | 67.6        |  68.0       |
    | 16           | 0.1      | 32         |  O1        | 68.1        |  68.3       |
    | 16           | 0.11     | 32         |  O1        | 67.9        |  68.4       |
    | 16           | 0.12     | 32         |  O1        | 67.0        |  67.5       |
    | 16           | 0.1      | 32         |  O2        | 66.5        |  67.0       |
    | 16           | 0.1      | 128        |  O1        | 67.0        |  67.4       |

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
