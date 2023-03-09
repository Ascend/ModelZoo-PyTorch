# BMN for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

时间动作建议生成是一项具有挑战性和前景的任务，旨在定位真实世界视频中可能发生动作或事件的时间区域。BMN利用边界匹配（BM）机制来评估密集分布提案的置信度分数，该机制将提案作为起始和结束边界的匹配对，并将所有密集分布的BM对组合到BM置信图中，该方法同时生成具有精确时间边界和可靠置信分数的提名。

- 参考实现：

  ```
  url=https://github.com/JJBOY/BMN-Boundary-Matching-Network
  commit_id=a92c1d79c19d88b1d57b5abfae5a0be33f3002eb
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/video
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.2.2.post3；pillow==8.4.0 |
  | PyTorch 1.8 | torchvision==0.9.1；pillow==9.1.0 |
  
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

   请用户自行下载数据集**ActivityNet**，该数据集非常庞大，实际复现时使用已经提取好的特征数据集，可在源码实现链接上获取数据集的下载方式。

   数据集目录结构参考如下所示。

   ```
   ├── csv_mean_100
         ├── 视频1的特征csv
         ├── 视频2的特征csv
         │   ...             
         ├── 视频19228的特征csv
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
   主要参数：
   --data_path                         //数据集路径
   --finetune                          //是否微调
   --training_lr                       //学习率
   --weight_decay                      //权重衰减
   --train_epochs                      //训练轮数
   --batch_size                        //训练批次大小
   --data_path                         //数据路径
   --loss_scale                        //loss scale大小
   --opt-level                         //混合精度类型
   ```


# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@100 |  FPS | Epochs | AMP_Type | Torch_Version |
| :-----: | :-----: | :--: | :----: | :------: | :-----: |
| 1p-NPU  | - |  58.96 | 1    |       O1 |1.8    |
| 8p-NPU  | 75 | 525.94 | 10    |       O1 |1.8    |


# 版本说明

## 变更

2023.03.09：更新readme，重新发布。

2022.02.14：首次发布。

## FAQ

无。