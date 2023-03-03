# DnCNN for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

DnCNN使用端到端的神经网络模型进行图像降噪，DnCNN结合了ResNet的residual learning（残差学习），但不同的是，DnCNN并非是每隔两层就加一个shortcut，而是将网络的输出直接改成residual image（残差图片），DnCNN的优化目标是真实残差图片与网络输出之间的MSE（均方误差）。DnCNN强调了residual learning 和 batch normalization（批量归一化）在图像复原中相辅相成的作用，在较深的网络中，也能很快的收敛并获得很好的性能。此外，DnCNN可以用单模型应对不同程度的高斯去噪、具有多个放大因子的单图像超分辨率，以及具有不同质量因子的 JPEG 图像去块。

- 参考实现：

  ```
  url=https://github.com/SaoYan/DnCNN-PyTorch
  commit_id=6b0804951484eadb7f1ea24e8e5c9ede9bea485b
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

1. 获取数据集。

   用户需要将[参考实现](https://github.com/SaoYan/DnCNN-PyTorch)下面的 `data` 目录整个拷贝到当前源码包(DnCNN)根目录下或服务器的任意目录下。

   数据集目录结构参考如下所示。
   ```
   ├── data
        ├──Set12
            │──图片1
            │──图片2
            │   ...       
        ├──Set68  
            │──图片1
            │──图片2
            │   ...
        ├──train
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
     bash ./test/train_eval_8p.sh --data_path=/data/xxx/    # 启动评测脚本前，需对应修改评测脚本中的resume参数，指定ckpt文件路径
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。
   
   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --data_path                         //数据集路径
   --num_of_layers                     //网络模型层数
   --mode                              //训练模式
   --noiseL                            //噪声级别
   --val_noiseL                        //验证噪声级别      
   --epochs                            //重复训练次数
   --batchSize                         //训练批次大小
   --lr                                //初始学习率，默认：0.001
   --preprocess                        //创建h5py数据集
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V | 32.06 | 2200 |  50    |    O2    |      1.5      |
| 8p-竞品V | 31.35 |  -   |  50    |    O2    |      1.5      |
|  1p-NPU  | 31.85 | 10520 |  50    |    O2    |      1.8      |
|  8p-NPU  | 31.12 | 32100 |  50    |    O2    |      1.8      |

# 版本说明

## 变更

2022.03.18：首次发布。

## FAQ

无。

