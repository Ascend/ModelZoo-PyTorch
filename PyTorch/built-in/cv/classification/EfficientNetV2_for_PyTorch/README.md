# EfficientNetV2 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

EfficientNetV2是Efficient的改进版，accuracy达到了发布时的SOTA水平，而且训练速度更快参数来更少。相对EfficientNetV1系列只关注准确率，参数量以及FLOPs，V2版本更加关注模型的实际训练速度。


- 参考实现：

  ```
  url=https://github.com/rwightman/pytorch-image-models/blob/v0.5.4/timm/models/efficientnet.py
  commit_id=9ca343717826578b0e003e78b694361621c2b0ef
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
  | PyTorch 1.5 | pillow==8.4.0 |
  | PyTorch 1.8 | pillow==9.1.0 |
  
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

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，CIFAR-10等，将数据集上传到服务器任意路径下并解压。

   EfficientNetV2迁移使用到的ImageNet2012数据集目录结构参考如下所示。

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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

--data_path参数填写数据集路径，需写到数据集的一级目录。

模型训练脚本参数说明如下。

   ```
公共参数：
--momentum                           //动量
--weight-decay                       //权重衰减
--lr                                 //初始学习率
--epochs                             //训练周期数
--seed                               //随机数种子设置
--workers                            //训练进程数
--model                              //训练模型名
--batch-size                         //训练批次大小
--apex-amp                           //使用apex进行混合精度训练（默认）
--native-amp                         //使用amp进行混合精度训练，启用时须将--apex-amp删除
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V |   -   | 533  |   1    |    O1    |      1.8      |
| 8p-竞品V | 82.34 | 4100 |  350   |    O1    |      1.8      |
|  1p-NPU-ARM  |   -   | 602  |   1    |    O1    |      1.8      |
|  8p-NPU-ARM  | 82.19 | 4100 |  350   |    O1    |      1.8      |
|  1p-NPU-非ARM  |   -   | 602  |   1    |    O1    |      1.8      |
|  8p-NPU-非ARM  | 82.19 | 4500 |  350   |    O1    |      1.8      |


# 版本说明

## 变更

2023.02.21：更新readme，重新发布。

2022.10.14：首次发布。

## FAQ

无。