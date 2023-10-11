# DynamicUNet for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

DynamicUNet 模型是一个图像分割任务上的SOTA模型，它赢得了许多Kaggle比赛，
该模型在工业中取得了广泛应用，该模型使我们能够在像素级别精确地对图像的每个部分进行分类。

- 参考实现：

  ```
  dynamic_unet:
   url=https://github.com/fastai/fastai/blob/master/fastai/vision/models/unet.py
   commit_id=7ec403cd41079bc81d80d48de67f7ab2b8141929
  awesome-semantic-segmentation-pytorch:
   url=https://github.com/Tramac/awesome-semantic-segmentation-pytorch
   commit_id=9d9e25da10e2299cf0c84b6e0be1c49085565d22  
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

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.6.0 |
  | PyTorch 1.8 | torchvision==0.9.1 |
  | PyTorch 1.11 | torchvision==0.12.0 |
  | PyTorch 2.1 | torchvision==0.16.0 |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本

  pip install -r 1.11_requirements.txt  # PyTorch1.11版本

  pip install -r 2.1_requirements.txt  # PyTorch2.1版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集 `VOC2012` ，将下载好的VOCdevkit数据集解压放置在源码包根目录下或者软链到源码包根目录下。 
   数据集目录结构参考如下所示。

   ```
    VOCdevkit
        └── VOC2012
            ├── Annotations
            ├── ImageSets
            │   └── Segmentation
            ├── JPEGImages
            ├── SegmentationObject
            └── SegmentationClass              
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

## 获取预训练模型
该模型的训练需要 `resnet50-19c8e357.pth` 预训练模型，在训练过程中其会自动下载，若存在网络等问题无法在训练时下载，请手动下载并放置于任意目录下，且在训练时需指定 `--more_path1=path/to/resnet50` 。预训练模型目录结构参考如下：
   ```
   path/to/resnet50
            └── resnet50-19c8e357.pth
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
     bash ./test/train_performance_1p.sh --more_path1=path/to/resnet50 # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --more_path1=path/to/resnet50 # 8卡精度
     bash ./test/train_performance_8p.sh --more_path1=path/to/resnet50 # 8卡性能   
     ```

   **注：more_path1为可选参数，用于指定预训练模型位置。**

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --dataset-path                      //数据集路径
   --worker                            //加载数据进程数      
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小，默认：240
   --lr                                //初始学习率，默认：1
   --momentum                          //动量，默认：0.9
   --weight_decay                      //权重衰减，默认：4e-5
   --amp                               //是否使用混合精度
   ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| Name  | mIoU   | FPS  | Epochs  | AMP_Type | Torch_Version |
|:-----:|:------:|:----:|:-------:| :-------:| :------------:|
| 1P-竞品V | -     | -    | -       | O1       | 1.5 |
| 8P-竞品V | -     | -    | 50      | O1       | 1.5 |
| 1P-NPU | -     | 7.29  | -       | O1       | 1.8 |
| 8P-NPU | 0.521 | 52.76   | 50      | O1       | 1.8 |

# 版本说明

## 变更

2022.11.26：更新Readme。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md

