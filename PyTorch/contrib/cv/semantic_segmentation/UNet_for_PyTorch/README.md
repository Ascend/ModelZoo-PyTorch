# UNet for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述
UNet在生物医学图像分割领域，得到了广泛的应用。
它是完全对称的，且对解码器（应该自Hinton提出编码器、解码器的概念来，
即将图像->高语义feature map的过程看成编码器，高语义->像素级别的分类score map的过程看作解码器）
进行了加卷积加深处理。

- 参考实现：

  ```
  url=https://github.com/4uiiurz1/pytorch-nested-unet
  commit_id=557ea02f0b5d45ec171aae2282d2cd21562a633e
  ```
- 适配昇腾 AI 处理器的实现：
  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/semantic_segmentation
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.2.2.post3 |
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

  pip install -r 1.11_requirements.txt # PyTorch1.11版本

  pip install -r 2.1_requirements.txt  # PyTorch2.1版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。


## 准备数据集

1. 用户自行获取 `data-science-bowl-2018` 数据集。
2. 上传数据集在源码包根目录下新建的 `inputs` 文件夹下并解压。
3. 数据集需要执行预处理，在源码包根目录下执行
    ```bash
    python3 preprocess_dsb2018.py
    ```
   数据集目录结构参考如下所示。
    ```
    inputs
    └── data-science-bowl-2018
        ├── stage1_train
        |   ├── 00ae65...
        │   │   ├── images
        │   │   │   └── 00ae65...
        │   │   └── masks
        │   │       └── 00ae65...            
        │   ├── ...
        |
    ...
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
     bash ./test/train_full_1p.sh --data_path={data/path} # 单卡精度
     bash ./test/train_performance_1p.sh --data_path={data/path} # 单卡性能     
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path={data/path} # 8卡精度
     bash ./test/train_performance_8p.sh --data_path={data/path} # 8卡性能
     ```

   - 单机8卡评测
  
     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh --data_path={data/path} # 8卡评测
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --device_id             //训练设备卡号
   --data_path             //数据集路径
   --optimizer             //优化器
   --epochs                //训练重复次数
   --batch_size            //训练批次大小
   --lr                    //初始学习率
   多卡训练参数：
   --device                //训练设备
   --num_gpus              //训练设备数量
   --rank_id               //训练设备卡号
   ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2** 训练结果展示表

| NAME   | Acc    |  FPS | Epochs | AMP_Type | Torch_Version |
|:------:|:------:|:----:|:------:|:--------:| :-----------: |
| 1p-竞品V | -     |  823 | 1      |        - | 1.5 |
| 8p-竞品V | 83.91 | 2332 | 100    |        - | 1.5 |
| 1p-NPU | -     |  231.15 | 1      |       O2 | 1.8 |
| 8p-NPU | 83.31 |  2070.295 | 100    |       O2 | 1.8 |

# 版本说明

## 变更

2022.12.20：整改readme，重新发布。

## FAQ

无。
