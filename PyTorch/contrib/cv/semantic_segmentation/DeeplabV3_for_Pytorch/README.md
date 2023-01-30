# DeeplabV3 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

DeepLabV3是一个经典的语义分割网络，采用空洞卷积来代替池化解决分辨率的下降（由下采样导致），采用ASPP模型实现多尺度特征图融合，提出了更通用的框架，适用于更多网络。

- 参考实现：

  ```
  url=https://github.com/fregu856/deeplabv3
  commit_id=415d983ec8a3e4ab6977b316d8f553371a415739
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/semantic_segmentation
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

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 硬件 | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | NPU固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/) |
  | mmcv  | 1.3.9 |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  cd ${code_path}

  bash env_set.sh
  ```

  如在bash env_set.sh过程中出现无法连接mmcv的github报错，可以参考env_set.sh脚本中的配置命令进行mmcv的下载、编译以及替换操作。其中mmcv替换操作需要指定其安装目录，具体操作如下所示。
  ```
  mmcv_path=mmcv安装路径
  ```

  ```
  cd ${code_path}

  /bin/cp -f mmcv_need/_functions.py ${mmcv_path}/mmcv/parallel/
  /bin/cp -f mmcv_need/scatter_gather.py ${mmcv_path}/mmcv/parallel/
  /bin/cp -f mmcv_need/distributed.py ${mmcv_path}/mmcv/parallel/
  /bin/cp -f mmcv_need/dist_utils.py ${mmcv_path}/mmcv/runner/
  ```
## 准备数据集

1. 获取数据集。

- 下载cityscapes数据集

- 新建文件夹data

- 将cityscas数据集放于data目录下

   ```shell
   ln -s /path/to/cityscapes/ ./data
   # 注：'/path/to/cityscapes/'为数据集存放的路径，根据实际路径进行指定。
   ```
- 配置数据集路径

  ```
  vim configs/_base_/datasets/cityscapes.py
  ```
  修改19行data_root为data文件夹路径

2. 数据预处理。

- 执行以下命令进行数据预处理操作：

   ```shell
   python3 tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
   # 注：'data/cityscapes'为数据集存放的路径，根据实际路径进行指定。
   ```
- 预处理后数据集目录结构参考如下所示。

  ```
  ├data
  ├─ cityscapes   
  ├── gtFine
  │       ├── test     
  │       │       ├──城市1──图片1、2、3、4
  │       │       ├──城市2──图片1、2、3、4  
  │       ├── train
  │       │       │──城市3──图片1、2、3、4
  │       │       ├──城市4──图片1、2、3、4  
  │       └── val      
  │       │       │──城市5──图片1、2、3、4
  │       │       ├──城市6──图片1、2、3、4  
  ├── leftImg8bit
  │       ├── test     
  │       │       ├──城市1──图片1、2、3、4
  │       │       ├──城市2──图片1、2、3、4  
  │       ├── train
  │       │       │──城市3──图片1、2、3、4
  │       │       ├──城市4──图片1、2、3、4  
  │       └── val      
  │       │       │──城市5──图片1、2、3、4
  │       │       ├──城市6──图片1、2、3、4
  ```

## 获取预训练模型

若无法自动下载，可手动下载resnet50_v1c.pth，并放到/root/.cache/torch/checkpoints/文件夹下。

# 开始训练

## 训练模型

1. 进入模型代码所在路径。

   ```
   cd /${code_path} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=real_data_path
     ```

   - 单机单卡性能

     启动单卡性能测试。

     ```
     bash ./test/train_performance_1p.sh --data_path=real_data_path
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=real_data_path
     ```

   - 单机8卡性能

     启动8卡性能测试。
     
     ```
     bash ./test/train_performance_8p.sh --data_path=real_data_path
     ```

   --data_path参数填写数据集路径。
   
   训练完成后，权重文件保存在当前路径的output文件夹下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | mIOU |  FPS | Epochs | AMP_Type |
| ------- | ----- | ---: | ------ | -------: |
| 1p-torch1.5 | 90.97   |  6.657 | 1000     |        - |
| 1p-torch1.8  | 91.11     |  6.806 | 1000      |       O2 |
| 8p-torch1.5 | 94.49 | 36.689  | 1000    |        - |
| 8p-torch1.8  | 96.13 | 38.925 | 1000    |       O2 |

# 版本说明

## 变更

2022.08.31：更新内容，重新发布。

2020.07.08：首次发布。

## 已知问题

无。











