# DB++ for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述

## 简述

#### mmocr

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmocr
  commit_id=26bc4713d4a451ed510a67be0a4fdd9903fd9011
  ```
# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套        | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动   | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=community) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/community) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)|
  | Apex | [0.1](https://gitee.com/ascend/apex/tree/master/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖（根据模型需求，按需添加所需依赖）。
- 首先卸载已安装的mmcv, mmcv-full, mmdet, mmocr

  ```
  pip install -r requirements.txt
  pip install mmcv-full -f http://download.openmmlab.com/mmcv/dist/npu/torch1.8.0/index.html
  pip install mmdet
  ```
- 请注意在x86下开启cpu性能模式 [将cpu设置为performance模式](https://gitee.com/ascend/pytorch/blob/master/docs/zh/PyTorch%E8%AE%AD%E7%BB%83%E8%B0%83%E4%BC%98&%E5%B7%A5%E5%85%B7%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97/PyTorch%E8%AE%AD%E7%BB%83%E8%B0%83%E4%BC%98&%E5%B7%A5%E5%85%B7%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md#%E5%B0%86cpu%E8%AE%BE%E7%BD%AE%E4%B8%BAperformance%E6%A8%A1%E5%BC%8F)
  


## 准备数据集

1. 获取数据集。

   主要参考[mmocr-idcar2015](https://mmocr.readthedocs.io/en/latest/datasets/det.html?highlight=icdar)进行icdar2015数据集准备。
   准备好的icdar2015数据集为$data_path。

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理

    - 本模型不涉及
  
- $data_path 目录结构如下：
 ```
     $data_path
    ├── annotations
    ├── imgs
    ├── instances_test.json
    └── instances_training.json
```
## 获取预训练模型（必选）

- 需下载[synthtext预训练模型](https://download.openmmlab.com/mmocr/textdet/dbnet/dbnetpp_r50dcnv2_fpnc_100k_iter_synthtext-20220502-db297554.pth)，将下载好的文件move到$project/checkpoints/textdet/dbnetpp/res50dcnv2_synthtext.pth。



# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机8卡，单机单卡。

   - 单机单卡训练

     启动单卡训练

     ```
     bash ./test/train_full_1p.sh  --data_path=$data_path
     ```
     ```
     bash ./test/train_performance_1p.sh  --data_path=$data_path
     ```
    
     训练完成后，输出模型训练精度和性能信息。

   - 单机8卡训练

     启动8卡训练

     ```
     bash ./test/train_full_8p.sh  --data_path=$data_path
     ```
     ```
     bash ./test/train_performance_8p.sh  --data_path=$data_path
     ```
    
     训练完成后，输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

| NAME     | Accuracy-Highest |  samples/s | AMP_Type |
| -------  | -----  | ---: | -------: |
| 1p-竞品A  | Hmean: 0.861 | 26.08 |       O1 |
| 1p-NPU   | Hmean: 0.861 | 15.3 |       O1 |
| 8p-竞品A  | Hmean: 0.861 | 186.4 |       O1 |
| 8p-NPU   | Hmean: 0.861 | 115.66 |       O1 |

备注：注意cpu设置为performance模式。

# 版本说明

## 变更

2022.11.7：首次发布

## 已知问题

**_当前发行版本中存在的问题描述。_**

无。
