# RDN for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述

## 简述

RDN主要是提出了网络结构RDB(residual dense blocks)，它本质上就是残差网络结构与密集网络结构的结合。RDN是针对图像复原任务的CNN模型。包含四个模块：Shallow feature extraction net（SFENet）表示前两个卷积层，用于提取浅层特征；Residual dense blocks（RDBs）融合残差模块和密集模块，每个块还包含Local feature fusion 和Local residual learning；Dense feature fusion（DFF）包含Global feature fusion 和Global residual learning 两部分；Up-sampling net（UPNet）网络最后的上采样（超分任务需要）+卷积操作。

- 参考实现：

  ```
  url=https://github.com/yjn870/RDN-pytorch
  commit_id=f2641c0817f9e1f72acd961e3ebf42c89778a054
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/others
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

   请用户自行准备数据集，包含训练集和验证集两部分，训练集使用DIV2K，验证集使用Set5。将准备好的训练集和验证集上传至服务器任意目录下并解压，解压后的训练集和验证集目录结构参考如下所示。

   ```
   ├── data
         ├──DIV2K_x2.h5
         ├──Set5_x2.h5                 
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
   --data_path                         //数据集路径
   --device_id                         //设置训练用卡ID
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小
   --lr                                //初始学习率
   --weight_decay                      //权重衰减
   --seed      					    //随机数种子设置
   --loss_scale_value                  //混合精度loss scale大小
   --apex_opt_level                    //混合精度类型
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1  | FPS      | Epochs | AMP_Type | Torch_version |
| :-----: | :-----: | :------: | :----: | :------: | :----------: |
| 1p-NPU  | -      | 544      | 1      | O1       | 1.8           |
| 8p-NPU  | 37.97  | 4337     | 800    | O1       | 1.8           |

# 版本说明

## 变更

2023.03.15: 更新readme，重新发布。

2022.01.30：首次发布。

## FAQ

无。