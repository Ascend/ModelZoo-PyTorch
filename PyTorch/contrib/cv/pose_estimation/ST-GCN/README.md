# ST-GCN for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述
动态骨架模型ST-GCN，它可以从数据中自动地学习空间和时间的patterns，这使得模型具有很强的表达能力和泛化能力。在Kinetics和NTU-RGBD两个数据集上与主流方法相比，取得了质的提升。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmskeleton/tree/master/deprecated/origin_stgcn_repo
  commit_id=b4c076baa9e02e69b5876c49fa7c509866d902c7
  ```
- 适配昇腾 AI 处理器的实现：
  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/pose_estimation
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

   用户自行下载 `Kinetics-skeleton` 数据集，将数据集上传到服务器任意路径下并解压。
   
   数据集目录结构参考如下所示。
   
   ```
   ├── Kinetics
         ├──kinetics-skeleton
              ├──train_data.npy     
              ├──train_label.pkl
              ├──val_data.npy
              ├──val_label.pkl 
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

   - 单机单卡评测

     启动单卡评测

     ```
     bash ./test/train_eval.sh --data_path={data/path} #单卡评测
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   --device                设备卡号
   --data_path             数据集路径
   --use_gpu_npu           使用的设备，npu或gpu
   --lr                    学习率
   --train_epochs          训练epoch
   --test_path_dir    		模型保存路径
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示


**表 2** 训练结果展示表
    
| NAME      | Acc@1 |     FPS | Epochs | AMP_Type | Torch_Version |
| :-------: | :-----: | :------: | :------: | :-------: | :----: |
| 1P-竞品V | -     | - | 2      |  - | 1.5 |
| 8P-竞品V | -     | - | 50      |  - | 1.5 |
| 1P-NPU | -     | 490.122 | 2      |       O2 | 1.8 |
| 8P-NPU | 31.75 | 1289.94 | 50     |       O2 | 1.8 |


# 版本说明

## 变更

2022.11.14：更新内容，重新发布。

## FAQ

无。


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
