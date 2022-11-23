# Resnet50-cifar for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)



# 概述

## 简述

MMClassification 是一款基于 PyTorch 的开源图像分类工具箱，是 OpenMMLab 项目的成员之一.
    主要特性
        支持多样的主干网络与预训练模型
        支持配置多种训练技巧
        大量的训练配置文件
        高效率和高可扩展性
        功能强大的工具箱

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmclassification
  commit_id=7b45eb10cdeeec14d01c656f100f3c6edde04ddd
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url} # 克隆仓库的代码
  cd {code_path} # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
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
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。
  ```
  1. pip3.7 install -r requirements.txt
  2. 安装mmcv
     cd /${模型文件夹名称}
     git clone --depth=1 https://github.com/open-mmlab/mmcv.git
     cd mmcv
     MMCV_WITH_OPS=1 pip3 install -e .
  3. 安装mmcls
     cd /${模型文件夹名称}
     pip3 install -e .
  ```


## 准备数据集

1. 获取数据集。

  模型训练所需要的数据集（cifar100）脚本会自动下载,请保持网络畅通.如果需要可用如下命令自行下载
    wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

   ```
   数据集目录结构参考如下所示:
   ```
    ├── cifar-100-python
      ├──file.txt   
      ├──train                  
      ├──meta     
      ├──test

 模型训练所需要的数据集（cifar100）脚本会自动下载,请保持网络畅通

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
     bash ./test/train_performance_1p.sh  # 1p性能
     bash ./test/train_full_1p.sh         # 1p精度 
     ```

   - 单机8卡训练

     启动8卡训练。
     ```
     bash ./test/train_performance_8p.sh   # 8p性能
     bash ./test/train_full_8p.sh          # 8p精度 
     ```

    注意：模型训练所需要的数据集（cifar100）脚本会自动下载,请保持网络畅通，如果已有数据集，这也可用传参的方式传入，例如：
      bash ./test/train_full_1p.sh --data_path=cifa100数据集路径

   

# 训练结果展示

**表 2**  训练结果展示表

| Acc@1  | FPS  | Npu_nums | Epochs | AMP_Type | Torch |
| :----: | :--: | :------: | :----: | :------: | :---: |
|   -    | 4196   |    1     |  2   |    O2    |  1.8  | 
| 61.65  | 32507  |    8     | 200  |    O2    |  1.8  |









