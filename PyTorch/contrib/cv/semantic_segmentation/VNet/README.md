# VNet for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

V-Net是一个早期的全卷积的三维图像分割网络，基本网络架构与2D图像分割网络U-Net相似，为了处理3D医学图像，采用了3D卷积模块和3D转置卷积模块。

- 参考实现：

  ```
  url=https://github.com/mattmacy/vnet.pytorch
  commit_id=a00c8ea16bcaea2bddf73b2bf506796f70077687
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

   请用户自行获取原始数据集，可选用开源数据集**LUNA16**，将数据集上传到服务器任意路径下并解压。

   以LUNA16数据集为例，数据集目录结构参考如下所示。

   ```
   ├── LUNA16
         ├──lung_ct_image
              ├──1.3.6.1.4.1.14519.5.2.1.6279.6001.997611074084993415992563148335.mhd                    
              ├──...                     
         ├──seg-lungs-LUNA16
              ├──1.3.6.1.4.1.14519.5.2.1.6279.6001.997611074084993415992563148335.mhd
              ├──...    
         ├──normalized_lung_ct
              ├──1.3.6.1.4.1.14519.5.2.1.6279.6001.997611074084993415992563148335.mhd                    
              ├──...                     
         ├──normalized_lung_mask
              ├──1.3.6.1.4.1.14519.5.2.1.6279.6001.997611074084993415992563148335.mhd
              ├──...                                      
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理。
    ```
    cd /${模型文件夹名称}
    python3 normalize_dataset.py data_path vox_spacing Z_MAX Y_MAX X_MAX  # data_path参数填写实际数据集路径
    
    示例：python normalize_dataset.py /data/xxx/LUNA16 2.5 128 160 160
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
   --data                              //数据集路径
   --seed                              //随机种子
   --workers                           //加载数据进程数      
   --lr                                //初始学习率
   --lr_decay                          //学习率衰减
   --weight_decay                      //权重衰减
   --device                            //设备，默认：'npu'
   --nEpochs                           //重复训练次数
   --batchSz                           //训练批次大小
   --amp                               //是否使用混合精度
   --loss-scale                        //混合精度loss scale大小
   --opt-level                         //混合精度类型
   --device_id                         //设置训练卡ID
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Error rate |  FPS | Epochs | AMP_Type | Torch_Version |
| :-------: | :----: | :---: | :------: | :-------: | :-------: |
| 1p-NPU  | -   | 38.79 | 1      |       O2 |    1.8 |
| 8p-NPU  | 0.428% | 218.50 | 200    |       O2 |    1.8 |


# 版本说明

## 变更

2023.03.15：更新readme，重新发布。

2020.07.08：首次发布。

## FAQ

无。