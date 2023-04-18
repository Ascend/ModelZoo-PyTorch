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

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |
  | PyTorch 1.11   | - |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。


- 安装依赖。

  在模型源码包根目录下执行命令，安装模型需要的依赖。

  ```
  cd /${code_path}

  bash env_set.sh
  ```

  如在bash env_set.sh过程中出现无法连接mmcv的github报错，可以参考env_set.sh脚本中的配置命令进行mmcv的下载、编译以及替换操作。其中mmcv替换操作需要指定其安装目录，具体操作如下所示。
  ```
  mmcv_path=mmcv安装路径
  ```

  ```
  cd /${code_path}

  cp -f mmcv_need/_functions.py ${mmcv_path}/mmcv/parallel/
  cp -f mmcv_need/scatter_gather.py ${mmcv_path}/mmcv/parallel/
  cp -f mmcv_need/distributed.py ${mmcv_path}/mmcv/parallel/
  cp -f mmcv_need/dist_utils.py ${mmcv_path}/mmcv/runner/
  ```
## 准备数据集

1. 获取数据集。

   请用户自行下载**cityscapes**数据集，在源码包根目录下新建文件夹data，并将cityscas数据集放于data目录下。

2. 数据预处理。

   - 执行以下命令进行数据预处理操作：

   ```shell
   python3 tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
   # 注：'data/cityscapes'为数据集存放的路径。
   ```
   - 预处理后数据集目录结构参考如下所示。

   ```
   ├ data
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
   │        │       ├──城市2──图片1、2、3、4  
   │        ├── train
   │       │       │──城市3──图片1、2、3、4
   │       │       ├──城市4──图片1、2、3、4  
   │       └── val      
   │       │       │──城市5──图片1、2、3、4
   │       │       ├──城市6──图片1、2、3、4
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

## 获取预训练模型

运行训练脚本会自动下载预训练模型，若无法自动下载，可手动下载resnet50_v1c.pth，并放到/root/.cache/torch/hub/checkpoints/文件夹下。

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
     bash ./test/train_full_1p.sh --data_path=./data/cityscapes/  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=./data/cityscapes/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=./data/cityscapes/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=./data/cityscapes/  # 8卡性能
     ```

   - 多机多卡性能训练
     
     ```
     1. 安装环境
     2. 开始训练，每个机器请按下面提示进行配置
     bash ./test/train_performance_multinodes.sh --data_path=数据集路径 --batch_size=单卡batch_size --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --seed                              //随机数种子设置
   --master-addr                       //主机地址
   --master-port                       //主机端口号
   --options                           //自定义选项    
   --gpu-ids                           //单卡训练卡号指定
   --load-from                         //权重加载
   --work-dir                          //日志和模型保存目录
   ```

   训练完成后，权重文件保存在当前路径的output文件夹下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | aACC | mIoU  |  FPS | Train_Step | AMP_Type | Torch_Version |
| :-----: | :---: | :--: | :----: | :------: | :------: | :------: |
| 1p-NPU | -    | -     | 8.78 | 1000      |       O2 | 1.8 |
| 8p-NPU | 96.13 | 78.98 | 75.135 | 7000 |       O2 |   1.8 |

# 版本说明

## 变更

2023.02.20：更新readme，重新发布。

2020.07.08：首次发布。

## FAQ

无。