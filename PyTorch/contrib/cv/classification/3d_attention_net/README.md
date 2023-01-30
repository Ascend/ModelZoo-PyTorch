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

  | 配套       | 版本                                                                          |
  |-----------------------------------------------------------------------------| ------------------------------------------------------------ |
  | 硬件       | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)                     |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip3.7 install -r requirements.txt
  ```


## 准备数据集

1. 获取CIFAR-10数据
   * 在源码包根目录下创建data目录。
   * 数据集获取方式请参考模型参考仓readme，参考仓：https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch.git。
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
     bash ./test/train_full_1p.sh --data_path=./data/cifar-10-batches-py

     bash ./test/train_performance_1p.sh --data_path=./data/cifar-10-batches-py
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=./data/cifar-10-batches-py

     bash ./test/train_performance_8p.sh --data_path=./data/cifar-10-batches-py
     ```

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


# 训练结果展示

**表 2**  训练结果展示表


| Top1 acc | FPS  | Epochs | AMP_Type | Device |  Bs  |
| ---- | -------- | -------- | -------- | ------ | -------- |
|  - | 1432 |   1  |  O2  | npu_1p | 512  |
|  85.48 | 9587 |  300 |  O2  | npu_8p | 512  |
|  - | 1396 |   1  |  O2  | gpu_1p | 512  |
|  85.37 | 8342 |  300 |  O2  | gpu_8p | 512  |
|  94.13 | 723  |  300 |  O2  | npu_8p |  32  |
|  94.32 | 2461 |  300 |  O2  | gpu_8p |  32  |


# 版本说明

## 变更

2023.1.30：整改Readme，重新发布。

## 已知问题

无。


