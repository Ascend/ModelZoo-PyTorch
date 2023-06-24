# DeCLIP for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

DeCLIP是一种数据高效的CLIP训练方法，通过利用图像-文本对之间的联系，DeCLIP可以更有效地学习通用视觉特征。
相较于CLIP需要4亿对图像-文本进行预训练， DeCLIP-ResNet50在使用更少的数据的同时在ImageNet上实现60.4%的准确度，
比CLIP-ResNet50高0.8%。

- 参考实现：

  ```
  url=https://github.com/Sense-GVT/DeCLIP
  commit_id=9d9e25da10e2299cf0c84b6e0be1c49085565d22
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/others
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.2.2.post3；pillow==8.4.0 |
  | PyTorch 1.8 | torchvision==0.9.1；pillow==9.1.0 |

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
  
- nltk_data准备(可选)
  - 该模型依赖nltk及其相关语料库(omw-1.4, stopwords, wordnet)
  - 若服务器不可连公网，则需要手动下载，放至 `~/nltk_data` 目录下。


## 准备数据集

1. 获取数据集。

   用户自行获取 `YFCC15M` 数据集。可参考 https://github.com/Sense-GVT/DeCLIP/blob/main/docs/dataset_prepare.md#prepare-datasets 进行数据集准备。并将数据集上传至源码包根目录下的 `dataset` 文件夹下，数据集目录结构参考如下所示。

   ```
   ├── dataset
         ├── yfcc15m_clean_open_data.json（约3.3G）               
         ├── yfcc15m_clean_open_data（依据yfcc15m_clean_open_data.json下载得到，约900G)
         ├── bpe_simple_vocab_16e6.txt.gz（约1.3M)
         ├── val_official.json（约5.6M)
         ├── imagenet_val（magenet valid数据集，需按照ILSVRC2012_val_********.JPEG的格式放在imagenet_valid文件夹内，不包含二级目录，约6.4G，5万张图片）
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
     bash ./test/train_full_1p.sh    # 单卡精度
     bash ./test/train_performance_1p.sh  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh  # 8卡精度
     bash ./test/train_performance_8p.sh  # 8卡性能   
     ```

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --config                            //训练配置
   ```
   
   训练完成后，权重文件保存在./checkpoint下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 |    FPS | Steps   | AMP_Type | Torch_Version |
|:-------:|:-----:|:------:| :-----: | :------: | :-----------: |
| 1p-竞品A  | -     |     85 | 1000    |       O1 | 1.5 |
| 8p-竞品A  | 24.7  |    560 | 128000  |       O1 | 1.5 |
| 1p-NPU  | -     | 143.26 | 1000    |       O1 | 1.8 |
| 8p-NPU  | 31.52 | 537.43 | 128000  |       O1 | 1.8 |
| 32p-NPU | 43.2  |  20000 | 128000  |       O1 | 1.8 |


# 版本说明

## 变更

2022.08.16：首次发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 ```./public_address_statement.md```









