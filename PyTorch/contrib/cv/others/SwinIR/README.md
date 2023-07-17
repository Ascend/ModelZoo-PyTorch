#  SwinIR for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

SwinIR是一个使用Swin转变的经典的图像复原网络。

- 参考实现：

  ```
  url=https://github.com/JingyunLiang/SwinIR.git 
  commit_id=9b1a9bf5d1df3b18c32a49ea82f60c313d779f7d
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git 
  code_path=PyTorch/contrib/cv/others
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}         # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括DIV2K，Set5等，将数据集上传到服务器任意路径下并解压。

   以DIV2K和Set5数据集为例，DIV2K数据集目录结构参考如下所示。

   ```
    ├── DIV2K 
    │    ├──DIV2K_test_LR_bicubic  
    │    │      ├──X2
    │    │      │   ├──图片1、2、3、4 ...
    │    │      ├──X3
    │    │      │   ├──图片1、2、3、4 ...
    │    │      ├──X4
    │    │      │   ├──图片1、2、3、4 ...    
    │    ├──DIV2K_test_LR_unknown 
    │    │      ├──X2
    │    │      │   ├──图片1、2、3、4 ...
    │    │      ├──X3
    │    │      │   ├──图片1、2、3、4 ...
    │    │      ├──X4
    │    │      │   ├──图片1、2、3、4 ...  
    │    ├──DIV2K_train_HR
    │    │      ├──图片1、2、3、4 ... 
    │    ├──DIV2K_train_LR_bicubic
    │    │      ├──X2
    │    │      │   ├──图片1、2、3、4 ...
    │    │      ├──X3
    │    │      │   ├──图片1、2、3、4 ...
    │    │      ├──X4
    │    │      │   ├──图片1、2、3、4 ...     
    │    ├──DIV2K_train_LR_unknown    
    │    │      ├──X2
    │    │      │   ├──图片1、2、3、4 ...
    │    │      ├──X3
    │    │      │   ├──图片1、2、3、4 ...
    │    │      ├──X4
    │    │      │   ├──图片1、2、3、4 ...     
   ```
   Set5数据集目录结构参考如下所示。

   ```
    ├── Set5 
    │    ├──GTmod12
    │    │      ├──图片1、2、3、4 ...
    │    ├──LRbicx2
    │    │      ├──图片1、2、3、4 ...
    │    ├──LRbicx3  
    │    │      ├──图片1、2、3、4 ...  
    │    ├──LRbicx4 
    │    │      ├──图片1、2、3、4 ...
    │    ├──original
    │    │      ├──图片1、2、3、4 ...   
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
     bash ./test/train_full_1p.sh --data_path1=./DIV2K --data_path2=./Set5    
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path1=./DIV2K --data_path2=./Set5 
     ```

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path1                              //训练数据集路径
   --data_path2                              //训练过程中用于测试模型精度的数据集路径
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 |  FPS | Epochs | AMP_Type |
| ------- | ----- | ---: | ------ | -------: |
| 1p-v100 | 38.23dB     |  15.09 | 300     |        - |
| 1p-910  | 37.62dB     |  8.88 | 300      |       O2 |
| 8p-v100 | 38.23dB | 109.58 | 300    |        - |
| 8p-910  | 37.63dB | 63.83 | 300    |       O2 |

# 版本说明

## 变更

2022.09.15：首次发布。

## 已知问题

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md




