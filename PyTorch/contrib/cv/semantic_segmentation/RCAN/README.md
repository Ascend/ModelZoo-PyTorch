# RCAN for PyTorch


-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)


# 概述

## 简述

卷积神经网络（CNN）深度对于图像超分辨率（SR）至关重要。然而，更深层的图像SR网络更难训练。低分辨率的输入和特征包含丰富的低频信息，这些信息在通道间被平等对待，因此阻碍了CNN的表征能力。为了解决这些问题，提出了超深剩余通道注意网络（RCAN）。具体地说，RCAN是一种残差中残差（RIR）结构来形成非常深的网络，它由几个具有长跳跃连接的残差组组成。每个剩余组包含一些具有短跳过连接的剩余块。同时，RIR允许通过多个跳转连接绕过丰富的低频信息，使主网络专注于学习高频信息。此外，提出了一种通道注意机制，通过考虑通道之间的相互依赖性，自适应地重新缩放通道特征。大量实验表明，RCAN相对于最先进的方法实现了更好的准确性和视觉改善。

- 参考实现：

    ```
    url=https://github.com/yjn870/RCAN-pytorch.git
    commit_id=0cba4c714eea8b2fdbc9a146313088cd9cc134f5
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


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，模型训练使用DIV2K数据集，请自行在 DIV2K dataset 官网上下载 DIV2K_train_HR.zip和DIV2K_valid_HR.zip，将数据集上传到服务器任意路径下并解压。模型评测使用Set5数据集，数据请用户自行获取。

   以DIV2K数据集为例，数据集目录结构参考如下所示。

    ```
   ├── DIV2K
         ├── train
              ├──图片1     
              ├──图片2  
              ├── ...                  
         ├── test  
              ├──图片1     
              ├──图片2  
              ├── ... 
   ├── Set5
         ├── original
              ├──图片1     
              ├──图片2  
              ├──图片3
              ├──图片4     
              ├──图片5   
    ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理。

    训练前需要进行数据集预处理，对原始数据集进行裁剪保存，请在源码包根目录下执行以下命令。

    ```
    python3 ./dataset_make.py --input_zip_path=raw_data_path --dataset_path=real_traindata_path

    # raw_data_path为下载的两个DIV2K压缩包所在的目录路径。
    # real_traindata_path为存储最终增强数据集的路径。

    示例：python3 ./dataset_make.py --input_zip_path=/home/dataset/dataset_RCAN/DIV2K/ --dataset_path=/home/dataset/dataset_RCAN/dataset_DIV2K/
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
     bash ./test/train_full_1p.sh --train_dataset_dir=real_traindata_path --test_dataset_dir=real_testdata_path  # 单卡精度
     
     bash ./test/train_performance_1p.sh --train_dataset_dir=real_traindata_path --test_dataset_dir=real_testdata_path  # 单卡性能
    
     示例：bash ./test/train_full_1p.sh --train_dataset_dir=/home/dataset/dataset_RCAN/dataset_DIV2K/ --test_dataset_dir=/home/dataset/dataset_RCAN/Set5/original/
     ```
   
   - 单机8卡训练
   
     启动8卡训练。
   
     ```
     bash ./test/train_full_8p.sh --train_dataset_dir=real_traindata_path --test_dataset_dir=real_testdata_path  # 8卡精度
     
     bash ./test/train_performance_8p.sh --train_dataset_dir=real_traindata_path --test_dataset_dir=real_testdata_path  # 8卡性能
     
     示例：bash ./test/train_full_8p.sh --train_dataset_dir=/home/dataset/dataset_RCAN/dataset_DIV2K/ --test_dataset_dir=/home/dataset/dataset_RCAN/Set5/original/
     ```

   --train_dataset_dir参数填写训练集路径，目录层级参考上述示例。

   --test_dataset_dir参数填写测试集路径，目录层级参考上述示例。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --arch                              //网络模型选择，默认："RCAN"
   --train_dataset_dir                 //训练数据文件路径
   --test_dataset_dir                  //测试数据文件路径
   --batch_size                        //训练批次大小
   --num_epochs                        //训练周期数
   --lr                                //初始学习率
   --workers                           //加载数据线程数
   --seed                              //随机数种子设置
   --amp                               //是否使用混合精度
   --loss_scale                        //设置loss scale大小
   --opt_level                         //混合精度类型
   --device_id                         //训练卡ID设置
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | PSNR |  FPS | Epochs | AMP_Type | Torch_version |
| :-----: |:---: | :--: | :----: | :-----: | :------: |
| 1p-竞品V | -   |  - | 1      |      O2 | 1.5 |
| 8p-竞品V | 38.12  | 828 | 600  |      O2 | 1.5 |
| 1p-NPU  | -   |280.168| 1      |       O2 | 1.8 |
| 8p-NPU  | 37.815 |1303.86 | 600    |      O2 | 1.8 |

# 版本说明

## 变更

2023.03.15：更新readme，重新发布。

2021.01.10：首次发布。

## FAQ
无。
