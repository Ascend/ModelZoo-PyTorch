# MobileNetv2 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述

## 简述

 MobileNetv2在MobileNetv1的基础上新提出了Linear Bottleneck 和 Inverted Residuals两种方法，使能在保持类似精度的条件下显著的减少模型参数和计算量。

- 参考实现：

  ```
  url=https://github.com/pytorch/vision.git 
  commit_id=7bf6e7b149720144be9745b2e406672d1da51957
  ```

- 适配昇腾 AI 处理器的实现：

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

   用户自行下载 `ImageNet` 数据集，将数据集上传到服务器任意路径下并解压。
   
   数据集目录结构参考如下所示。

   ```
   ├── ImageNet
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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/   # 单卡精度 
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/   # 8卡精度

     bash ./test/train_performance_8p.sh --data_path=/data/xxx/ # 8卡性能
     ```

   - 多机多卡性能数据获取流程

     ```
      1. 准备多机环境
      2. 开始训练，每个机器请按下面提示进行配置
          bash ./test/train_performance_multinodes.sh --data_path=数据集路径 --batch_size=单卡batch_size*单机卡数 --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
     ```
    --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --addr                    		// 节点ip
   --seed                     		// 随机数种子
   --workers                  		// 工作线程数
   --lr                       		// 学习率
   --print-freq               		// 多久一次打印
   --eval-freq 1              		// 多久一次验证
   --dist-url                 		// 分布式节点url
   --dist-backend             		// 分布式所用后台
   --multiprocessing-distributed   // 是否使用分布式
   --world-size                    // 总共进程数
   --class-nums                    //  类别数量
   --batch-size                    // 一次训练多少图片
   --epochs                        // 训练多少轮
   --rank                          // 当前进程排名
   --device-list                   //  使用哪些卡
   --amp                           // 混合精度
   --benchmark 0                   // 是否启用benchmark
   --data                          // 数据路径
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1  |    FPS  | Epochs | AMP_Type | Torch_Version |
| :-----: | :----: |  :---:  | :----: | :------: |  :-------:    |
| 1p-竞品V| - | - | 1  | - | 1.5 |
| 8p-竞品V| - | - | 600 | - | 1.5 |
| 1p-NPU(ARM)|   -    |   2789.13  |        |    O2    |   1.8    |
| 8p-NPU(ARM)| 71.3   | 13477.3 | 600    |    O2    |   1.8    |
| 1p-NPU(非ARM)|   -    |   2807.02  |        |    O2    |   1.8    |
| 8p-NPU(非ARM)| -   | 9190.41 | 600    |    O2    |   1.8    |


## 版本说明

## 变更

2022.10.24：更新torch1.8版本，重新发布。

2020.12.19：首次发布。

## FAQ

无。

