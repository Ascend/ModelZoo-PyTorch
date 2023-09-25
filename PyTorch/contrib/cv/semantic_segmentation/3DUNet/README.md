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

   用户自行获取原始数据集**MICCAI_BraTS_2018_Data**，将数据集上传到服务器任意路径下并解压。

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
   
   > **注意：**
   >数据集目录结构需保持一致，若已含有`generated`目录和`brats2018-list-train-samples-1024.txt/brats2018-list-val-samples-1024.txt`文件，请删除，否则会影响模型精度，训练脚本中会自动生成该目录和文件。


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
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```
     
   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录（示例：--data_path=/data/xxx/MICCAI_BraTS_2018_data）。
   
   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --data_path                         //数据集路径
   --workers                           //加载数据进程数
   --nEpochs                           //重复训练次数
   --batchSz                        	//训练批次大小
   --lr                                //初始学习率
   --amp                               //是否使用混合精度
   --device                            //设置设备类型
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   |  Dsc   |  FPS   | Epochs | AMP_Type | Torch_version |
| :------: | :----: | :----: | :----: | :------: | :-----------: |
| 1p-竞品V |   -    | 36.107 |   1    |    O1    |      1.5      |
| 8p-竞品V |  65.4  | 266.7  |  100   |    O1    |      1.5      |
|  1p-NPU  |   -    |   42   |   1    |    O1    |      1.8      |
|  8p-NPU  | 66.276 |  281   |  100   |    O1    |      1.8      |


# 版本说明

## 变更

2023.03.21：更新readme，重新发布。

2020.07.08：首次发布。

## FAQ

无。


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
