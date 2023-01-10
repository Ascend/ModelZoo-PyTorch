

# Gluon_ResNet50_v1b for PyTorch

-   [概述](#1)
-   [准备训练环境](#2)
-   [开始训练](#3)
-   [训练结果展示](#4)
-   [版本说明](#5)

# 概述

## 简述

ResNet是残差网络(Residual Network)的缩写，该系列网络广泛用于目标分类等领域以及作为计算机视觉任务主干经典神经网络的一部分，典型的网络有ResNet50, ResNet101等。ResNet网络证明网络能够向更深（包含更多隐藏层）的方向发展。

ResNet-b相比ResNet修改了下采样模块，将残差分支的下采样模块移动到3*3卷积中，避免信息的流失。

+ 参考实现：

  ```
  url=https://github.com/rwightman/pytorch-image-models
  commit_id=381b2797858248619fe8007fa1c5f5a5d4ab3919
  ```

+ 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification
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

  | 配套          | 版本                                                         |
  | ------------- | ------------------------------------------------------------ |
  | 硬件          | [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | NPU固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN          | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch       | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```
  pip install -r requirements.txt
  ```



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
     bash test/train_full_1p.sh  --data_path=real_data_path        # 1p精度
     bash test/train_performance_1p.sh --data_path=real_data_path  # 1p性能
     ```
   
   + 单机8卡训练

     启动8卡训练：
   
     ```
     bash test/train_full_8p.sh  --data_path=real_data_path        # 8p精度
     bash test/train_performance_8p.sh --data_path=real_data_path  # 8p性能 
     ```
   
     其中real_data_path参数填写数据集路径，例如/data/imagenet。
   
     模型训练脚本参数说明如下。
     
          公共参数：
          --data_path                //数据集路径
          --device_id                //训练使用的npu device卡id
          -b                         //batchsize
          --model                    //模型名称，默认值gluon_resnet50_v1b
          --local_rank               //卡id
          --amp                      //开启NVIDIA Apex AMP或者Native AMP
          --apex-amp                 //开启NVIDIA Apex AMP
          --sched                    //学习策略，默认值cosine
          --epochs                   //epoch数量
          --cooldown-epochs          //连续多少次epoch没有变化就进行学习率调整
          --lr                       //学习率
          --log-interval             //日志打印间隔
          --use-multi-epochs-loader  //是否使用multi-epochs-loader
     
     训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。
   

 


​	

# 训练结果展示

**表 2**  训练结果展示表

| Name   | Acc@1 | FPS      | Epochs | AMP_Type |
| ------ | :---: | -------- | :----: | :------: |
| 1p-1.5 |   -   | 1318.2   |   90   |    O2    |
| 1p-1.8 |   -   | 1626.62  |   90   |    O2    |
| 8p-1.5 | 77.22 | 9145.32  |   90   |    O2    |
| 8p-1.8 | 76.85 | 12204.45 |   90   |    O2    |



# 版本说明

## 变更

2022.08.17：首次发布。

2023.01.09：Readme整改。

## 已知问题

无。
