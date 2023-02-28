# 3d_attention_net for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

3d_attention_net是一个使用注意力机制的卷积神经网络，通过堆叠多个注意力模块Attention Module来构建，每个注意力模块包含两个分支：掩膜分支（mask branch）和主干分支(trunk branch)。其中主干分支可以是当前的任何一种SOTA卷积神经网络模型，掩膜分支通过对特征图的处理输出维度一致的注意力特质图（Attention Feature Map），然后使用点乘操作将两个分支特征图组合在一起，得到最终的输出特征图。它可以端到端的训练方式与最新的前馈网络结果相结合。
- 参考实现：

  ```
  url=https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch.git
  commit_id=88ed90f1b59f4b20e152495d3a5b6a19a4aa4232 
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

   用户自行获取 `CIFAR-10` 数据集，上传至服务器模型源码包根目录下新建的 `data` 文件夹下并解压。数据集获取方式请参考模型参考仓 `readme` 。

   数据集目录结构参考如下所示。
   ```
   |——data
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
     bash ./test/train_full_1p.sh --data_path=./data/cifar-10-batches-py  # 单卡精度

     bash ./test/train_performance_1p.sh --data_path=./data/cifar-10-batches-py # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=./data/cifar-10-batches-py  # 8卡精度

     bash ./test/train_performance_8p.sh --data_path=./data/cifar-10-batches-py # 8卡性能
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录，此模型默认为 `./data/cifar-10-batches-py` 。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --device_type                       //训练设配类型
   --device_id                         //训练指定用卡  
   --device_num                        //训练用卡数量
   --is_train                          //训练模式
   --is_pretrain                       //是否加载预训练模型
   --total_epochs                      //训练重复次数
   --train_batch_size                  //训练批次大小
   ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | Bs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :---: | :------: | :-----------: |
| 1p-竞品V |  -    | 1396 | 1   | 512 | O2 | 1.5 |
| 8p-竞品V | 85.37 | 8342 | 300 | 512 | O2 | 1.5 |
| 8p-竞品V | 94.32 | 2461 | 300 | 32  | O2 | 1.5 |
| 1p-NPU  |  -    | 1432 | 1   | 512 | O2 | 1.8 |
| 8p-NPU  | 85.48 | 9587 | 300 | 512 | O2 | 1.8 |
| 8p-NPU  | 94.13 | 723  | 300 | 32  | O2 | 1.8 |

# 版本说明

## 变更

2023.1.30：整改Readme，重新发布。

## FAQ

无。


