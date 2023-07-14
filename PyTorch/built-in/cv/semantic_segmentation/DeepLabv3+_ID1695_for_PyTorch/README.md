# Deeplabv3+ for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

DeepLabv3+是Google提出的DeepLab系列的第4代语义分割网络，此模型在DilatedFCN基础上引入了EcoderDecoder的思路，并配以空洞卷积和空间金字塔池化模块来提高分割精度，本仓库为Deeplabv3+的PyTorch实现。

- 参考实现：

  ```
  url=https://github.com/jfzhang95/pytorch-deeplab-xception.git
  commit_id=9135e104a7a51ea9effa9c6676a2fcffe6a6a2e6
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/semantic_segmentation
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version |          三方库依赖版本           |
  | :-----------: | :-------------------------------: |
  |  PyTorch 1.5  | torchvision==0.6.0；pillow==8.4.0 |
  |  PyTorch 1.8  | torchvision==0.9.1；pillow==9.1.0 |
  |  PyTorch 1.11 | numpy==1.21.6 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  ```
  > **说明：** 
  > 只需执行一条对应的PyTorch版本依赖安装命令。

## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，并将数据集上传到服务器任意路径下并解压。

   以VOC2012数据集为例，数据集目录结构参考如下所示。

   ```
    ├── VOC2012
    │   ├── Annotations
    │   ├── ImageSets
    │   ├── JPEGImages
    │   ├── SegmentationClass
    │   ├── SegmentationObject
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
   --dataset                           //数据集名称
   --workers                           //加载数据进程数      
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小
   --lr                                //初始学习率，默认：0.01
   --momentum                          //动量，默认：0.9
   --apex                              //是否使用混合精度
   --apex-opt-level                    //混合精度类型
   --device_id                         //指定训练卡id
   多卡训练参数：
   --multiprocessing_distributed       //是否使用多卡训练
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME | mIOU    | FPS       | Npu_nums | Epochs   | AMP_Type | Torch_Version |
| :------:| :------: | :------:  | :------: | :------: | :------: | :------: |
| 1p-NPU |   -   | 42.65 | 1        | 3      | O2     | 1.8 |
| 8p-NPU | 0.738 | 252.83 | 8        | 100      | O2   | 1.8 |

# 版本说明

## 变更

2020.02.16：更新readme，重新发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
