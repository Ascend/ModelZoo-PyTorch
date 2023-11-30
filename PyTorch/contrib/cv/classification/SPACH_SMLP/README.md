# SPACH-SMLP for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

SPACH-SMLP是一个经典的图像分类网络，结构简单。sMLP 模块通过稀疏连接和权重共享，显著降低了模型参数的数量和计算复杂度，避免了MLP模型性能的常见过拟合问题。

- 参考实现：

  ```
  url=https://github.com/microsoft/SPACH
  commit_id=b11b098970978677b7d33cc3424970152462032d
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
  | PyTorch 1.5 | torchvision==0.6.0 |
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

1. 获取数据集。
    用户自行获取原始数据集ImageNet2012，将数据集上传到服务器任意路径下并解压。数据集目录结构参考如下所示。
    
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

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh --data_path=/data/xxx/ --resume=ckpt_path
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。
   
   --resume参数填写模型训练生成的ckpt文件路径，写到文件的一级目录即可。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --batch-size                              //训练批次大小
   --epochs                                  //训练周期数
   --weight-decay                            //权重衰减
   --data-path                               //数据集路径
   --world_size                              //分布式训练进程数
   --npu                                     //使能npu训练
   --seed                                    //随机数种子设置
   --device                                  //训练设备
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。
   

# 训练结果展示

**表 2**  训练结果展示表

|   Name   | Acc@1  |   FPS   | Epochs | AMP_Type | Torch_Version |
| :------: | :----: | :-----: | :----: | :------: | :-----------: |
| 1p-竞品V |   -    | 198.94  |   1    |    O1    |      1.5      |
| 1p-竞品V | 81.70% | 1522.22 |  300   |    O1    |      1.5      |
|  1p-NPU  |   -    |   211   |   1    |    O1    |      1.8      |
|  8p-NPU  | 80.90% | 1628.5  |  300   |    O1    |      1.8      |

# 版本说明

## 变更

2023.02.23：更新readme，重新发布。

2022.09.15：首次发布。

## FAQ

无。


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md