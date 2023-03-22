# GENet for PyTorch

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


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.2.2.post3 |
  | PyTorch 1.8 | torchvision==0.9.1 |

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

1. 获取数据集
   用户自行获取 `CIFAR-10` 数据集，上传至服务器模型源码包根目录下新建的 `data` 目录下并解压。
   
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
     bash ./test/train_full_1p.sh --data_path=./data  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=./data # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=./data  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=./data # 8卡性能
     ```
  
   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh --data_path=./data  # 启动评测脚本前，需对应修改评测脚本中的--resume参数，指定ckpt文件路径
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --amp                               //是否使用混合精度
   --opt_level                         //混合精度训练级别
   --loss_scale                        //混合精度训练的损失尺度   
   --DataPath                          //数据集路径
   --rank                              //rank数量
   --dist-url                          //设置分布式训练网址
   --world-size                        //分布式训练节点数量
   --workers                           //数据加载进程
   --batch_size                        //训练批次大小
   --device                            //训练设备
   多卡训练参数：
   --multiprocessing-distributed       //使用分布式多卡训练
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V | 94.76 | 1350.074 |  300   |  O2  |    1.5    |
| 8p-竞品V | 94.81 | 6536.289 |  300   |  O2  |    1.5    |
|  1p-NPU  | 94.73 | 2900.000 |  300   |  O2  |    1.8    |
|  8p-NPU  | 95.23 | 16912.437 |  300  |  O2  |    1.8    |

# 版本说明

## 变更

2023.1.30：更新readme，重新发布。

## FAQ

无。