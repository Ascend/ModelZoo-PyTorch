# YOLOX for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

MMDetection 是一个基于 PyTorch 的目标检测开源工具箱。计算机视觉基础库 [MMCV](https://github.com/open-mmlab/mmcv)，MMCV 是 MMDetection 的主要依赖。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection
  commit_id=3e36d5cfd4fe7c550b4c3493360fd369b858b1dc
  
  配置文件：
  url=https://github.com/open-mmlab/mmdetection/blob/master/configs/yolox/yolox_m_8x8_300e_coco.py
  commit_id=2bdb1670e1f78930b0cd959263ed667ecada954d
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/dev/cv/detection
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

  | 配套        | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动   | [22.0.RC3](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.1.RC1](https://www.hiascend.com/software/cann/commercial?version=6.1.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)|

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖（根据模型需求，按需添加所需依赖）。

  ```
  pip install -r requirements.txt
  ```
  


## 准备数据集

1. 获取数据集。

   使用coco2017数据集。
   准备好数据集后放到 ./dataset 目录下

   ```
   ├── coco2017
         ├── annotations               
         	├── instances_train2017.json
         	├── instances_val2017.json ...
         ├── train2017
         	├── 000000******.jpg ...
         ├── val2017
         	├── 000000******.jpg ...
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理
    
    - 本模型不涉及

## 获取预训练模型（可选）

- 本模型不涉及

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
     cd test
     bash train_full_1p.sh --data_path=xx/xx/coco2017
  bash train_performance_1p.sh --data_path=xx/xx/coco2017
     ```

   - 单机8卡训练

     启动8卡训练。
   
     ```
  cd test
     bash train_full_8p.sh --data_path=xx/xx/coco2017 
  bash train_performance_8p.sh --data_path=xx/xx/coco2017
     ```
   
   训练完成后，pth文件保存在./work_dirs下面，权重文件保存在../test/output/$deviceid/ckpt下，并输出模型训练精度和性能信息。
   

# 训练结果展示

**表 2**  训练结果展示表

| NAME     | mAp(Iou=0.50:.95) |  FPS | Steps     |
| -------  | :---:  | ---: | :----:    |
| 8p-NPU   | -  | - | 1,479,200 |
| 8p-竞品A |       0.432       | 246.1 | 1,479,200 |

备注：一定要有竞品和NPU。

# 版本说明

## 变更

2022.11.29：首次发布

## 已知问题

**_当前发行版本中存在的问题描述。_**

无。











