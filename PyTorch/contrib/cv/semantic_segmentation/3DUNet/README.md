# 3DUnet for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

UNet是完全对称的，且对解码器（应该自Hinton提出编码器、解码器的概念来，即将图像->高语义feature map的过程看成编码器，高语义->像素级别的分类score map的过程看作解码器）进行了加卷积加深处理。3DUnet将所有2D操作替换为3D对应物，该实现执行动态弹性变形，以便在训练期间实现高效的数据增强。

- 参考实现：

  ```
  url=https://github.com/black0017/MedicalZooPytorch
  commit_id=6c21e28643a56e0924aa3de77145950183633d6f
  ```
  
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/semantic_segmentation
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}          # 克隆仓库的代码
  cd {code_path}     	  # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 硬件       | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial  ) |
  | 固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial   ) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1   ) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集:[[MICCAI_BraTS_2018_Data]](https://pan.baidu.com/s/1qELMb9M63bEevRBpr5QHyA )(提出码:02nx )，将数据集上传到服务器任意路径下并解压。

   数据集目录结构参考如下所示。

   ```
   ├── MICCAI_BraTS_2018_data
         ├──MICCAI_BraTS_2018_Data_Training
              ├──HGG
                    │──Brats18_2013_2_x
                    │──Brats18_2013_2_xx
                    │   ...       
              ├──LGG
                    │──Brats18_2013_0_x
                    │──Brats18_2013_0_xx
                    │   ...
              ├──survival_data.csv
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

   该模型支持单机8卡训练。

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/ 
     eg: bash ./test/train_full_8p.sh --data_path='/home/dataset/MICCAI_BraTS_2018_data'
     ```
   
   - 单机单卡性能
   
     启动单机性能
   
     ```
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/ 
     eg: bash ./test/train_performance_1p.sh --data_path='/home/dataset/MICCAI_BraTS_2018_data'
     ```
   
   - 单机8卡性能
   
     启动8卡性能
     
     ```
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/ 
     eg: bash ./test/train_performance_1p.sh --data_path='/home/dataset/MICCAI_BraTS_2018_data'
     ```
   
   --data_path参数填写数据集路径。
   
   
   
   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --data_path                         //数据集路径
   --workers                           //加载数据进程数
   --epoch                             //重复训练次数
   --batchSz                        	//训练批次大小
   --lr                                //初始学习率，默认：0.005
   --amp                               //是否使用混合精度
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME   | Dsc  | FPS  | Torch_version |
| ------ | ---- | ---- | ------------- |
| 1p-NPU | -    | 18   | Torch1.5      |
| 1p-NPU | -    | 42   | Torch1.8      |
| 8p-NPU | 66   | 148  | Torch1.5      |
| 8p-NPU | 70   | 281  | Torch1.8      |



# 版本说明

2022.11.24：更新pytorch1.8版本，重新发布。

2021.10.16：首次发布。

## 已知问题


无。











