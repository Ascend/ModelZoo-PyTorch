# GENet_for_PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

  GENet提出一种简单的、轻量级的方法，该方法能够在卷积神经网络中更好地利用上下文信息。该网络引入一对算子来实现这点：第一点即为收集，它有效的聚合了来自较大空间范围的特征响应；第二点即为激发这些聚合信息，将聚合信息重新分配为局部特征。不管是在所带来的参数数量的增加上还是在额外的计算复杂性方面，这些算子都很廉价，可以直接集成到架构中，并且可以提升性能。
- 参考实现：

  ```
  url=https://github.com/BayesWatch/pytorch-GENET
  commit_id=3fbf99fb6934186004ffb5ea5c0732e0e976d5b2
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
  | 硬件       | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```
  pip3.7 install -r requirements.txt
  ```


## 准备数据集


1. 获取CIFAR-10数据
   * 在源码包根目录下创建data目录。
   * 数据集获取方式请参考：http://www.cs.toronto.edu/~kriz/cifar.html 。
   * 将下载好的CIFAR-10数据集上传至data目录，而后解压。

    数据集目录结构参考如下所示。

    ```
    ├──cifar-10-batches-py
        ├── batches.meta            
        ├── data_batch_1   
        ├── data_batch_2
        ├── data_batch_3
        ├── data_batch_4
        ├── data_batch_5                         
        ├── readme.html
        └── test_batch           
    ```


   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。

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
     bash ./test/train_full_1p.sh --data_path=./data
     
     bash ./test/train_performance_1p.sh --data_path=./data
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=./data
     
     bash ./test/train_performance_8p.sh --data_path=./data
     ```

   data_path是数据集路径

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --amp                               //是否使用混合精度
   --opt_level                         //混合精度训练级别
   --loss_scale                        //混合精度训练的损失尺度   
   --DataPath                          //数据集路径
   --rank                              //rank数量
   --dist-url                          //设置分布式训练网址
   --multiprocessing-distributed       //使用分布式多卡训练
   --world-size                        //分布式训练节点数量
   ```

# 训练结果展示

**表 2**  训练结果展示表

| Acc@1 |    FPS    | Device Type | Device Nums | Epochs | AMP_Type |
|:-----:|:---------:|:-----------:|:-----------:|:------:|:--------:|
| 94.73 | 2900.000  |     NPU     |      1      |  300   |    O2    |
| 95.23 | 16912.437 |     NPU     |      8      |  300   |    O2    |
| 94.76 | 1350.074  |     GPU     |      1      |  300   |    O2    |
| 94.81 | 6536.289  |     GPU     |      8      |  300   |    O2    |

# 版本说明

## 变更

2023.1.30：更新readme，重新发布。


## 已知问题

无。