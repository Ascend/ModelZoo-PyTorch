# GoogLeNet for PyTorch

-   [概述](#1)
-   [准备训练环境](#2)
-   [开始训练](#3)
-   [训练结果展示](#4)
-   [版本说明](#5)

# 概述

## 简述

GoogLeNet是2014年Christian Szegedy提出的一种全新的深度学习结构，在这之前的AlexNet、VGG等结构都是通过增大网络的深度（层数）来获得更好的训练效果，但层数的增加会带来很多负作用，比如overfit、梯度消失、梯度爆炸等。inception的提出则从另一种角度来提升训练结果：能更高效的利用计算资源，在相同的计算量下能提取到更多的特征，从而提升训练结果。

+ 参考实现：

  ```
  url=https://github.com/pytorch/vision.git
  commit_id=35f68a09f94b2d7afb3f6adc2ba850216413f28e
  ```

+ 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.2.2.post3；pillow==8.4.0 |
  | PyTorch 1.8 | torchvision==0.9.1；pillow==9.1.0 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。


## 准备数据集

用户可以选用的数据集包括ImageNet2012、CIFAR-10等，本文档提供的训练脚本中，是以ImageNet2012数据集为例，数据集目录结构参考如下所示：

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
>
> 训练集和验证集图片分别位于train和val文件夹路径下，该目录下每个文件夹代表一个类别，同一文件夹下的所有图片都有相同的标签。
>
> 该数据集的训练脚本只作为一种参考示例，在使用其他数据集时，需要修改数据集路径。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   + 单机单卡训练

     启动单卡训练：

     ```
     bash test/train_full_1p.sh --data_path=real_data_path  # 单卡精度
     
     bash test/train_performance_1p.sh --data_path=real_data_path  # 单卡性能
     ```

   + 单机8卡训练

     启动8卡训练：

     ```
     bash test/train_full_8p.sh --data_path=real_data_path  # 8卡精度
     
     bash test/train_performance_8p.sh --data_path=real_data_path  # 8卡性能 
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data-path                //数据集路径
   --model                    //模型名称，默认值googlenet
   --epochs                   //训练周期数
   --batch-size               //训练批次大小
   --lr                       //学习率
   --momentum                 //动量值
   --weight-decay             //权重衰减值
   --apex                     //开启混合精度
   --apex-opt-level           //混合精度级别
   --loss_scale_value         //混合精度loss scale大小
   --seed                     //随机种子
   --device_id                //npu训练卡id
   --print-freq               //日志打印频率
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME  | Acc@1 | FPS | Epochs | AMP_Type | Torch_Version |
| ------ |:-----:|:---:|:------:|:--------:|:--------:|
| 1p-NPU |     - |   - |   1    |      O1 |      1.5 |
| 8p-NPU |     70.8984  |   1701.47  |     90 |  O1 |    1.5|


# 版本说明

## 变更

2023.02.21：更新readme，重新发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md