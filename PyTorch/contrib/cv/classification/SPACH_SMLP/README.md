# SPACH-SMLP for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

SPACH-SMLP是一个经典的图像分类网络，结构简单。sMLP 模块通过稀疏连接和权重共享，显著降低了模型参数的数量和计算复杂度，避免了MLP模型性能的常见过拟合问题。

- 参考实现：

  ```
  url=https://github.com/microsoft/SPACH
  commit_id=b11b098970978677b7d33cc3424970152462032d
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                                         |
  | ---------- | ---------------------------------------------------------------------------- |
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)                       |


- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。
  用户自行获取原始数据集ImageNet2012，将数据集上传到服务器任意路径下并解压。数据集目录结构参考如下所示。
   ```
   ├── ImageNet2012
         ├──train
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...   
              ├──...                     
         ├──val  
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...              
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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/    
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/   
     ```

   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                              //数据集路径
   ```
   
   
日志输出路径：

    test/output/devie_id/train_${device_id}.log           # training detail log

    test/output/devie_id/SPACH-SMLP_2_bs8192_8p_perf.log  # 8p training performance result log

    test/output/devie_id/SPACH-SMLP_2_bs8192_8p_acc.log   # 8p training accuracy result log

# 训练结果展示

**表 2**  训练结果展示表

| Name   | Acc@1  | FPS     | Epochs | AMP_Type |
| ------ | ------ | ------- | ------ | -------- |
| GPU-1p | -      | 198.94  | 1      | O1       |
| GPU-8p | 81.70% | 1522.22 | 300    | O1       |
| NPU-1p | -      | 211     | 1      | O1       |
| NPU-8p | 80.90% | 1628.5  | 300    | O1       |

# 版本说明

## 变更

2022.09.15：首次发布。

## 已知问题

无。











