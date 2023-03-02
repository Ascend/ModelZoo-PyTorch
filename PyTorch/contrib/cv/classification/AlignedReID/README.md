# AlignedReID for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

AlignedReID提取了图像中的全局特征并与局部特征联合学习，局部特征学习通过计算两组局部特征之间的最短路径来执行对齐/匹配而无需额外监督，在联合学习后只保留全局特征来计算图像之间的相似度。AlignedReID是第一个在market1501数据集上超越人类水平的ReID方法。

- 参考实现：

  ```
  url=https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch
  commit_id=2e2d45450d69a3a81e15d18fe85c2eebbde742e4 
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令。
  ```
  pip install -r requirements.txt
  ```

## 准备数据集

1. 获取数据集。

   用户自行获取 `market1501` 原始数据集，将数据集上传到服务器任意路径下并解压。

   数据集目录结构参考如下所示。

   ```
   ├── market1501
         ├──images
              ├──xxx.jpg
              |  ...
         ├──ori_to_new_im_name.pkl
         ├──partitions.pkl  
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
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh --data_path=/data/xxx/ --pth_path=real_pre_train_model_path # 8卡评测
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。
   
   --pth_path参数填写训练权重生成路径，需写到权重文件的一级目录。
   
   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --data_path                         //数据集路径
   --addr                              //主机地址
   --workers                           //加载数据进程数      
   --total_epochs                      //重复训练次数
   --ids_per_batch                     //每个设备训练批次大小
   --base_lr                           //初始学习率
   --exp_decay_at_epoch                //权重衰减epoch
   --npu                               //训练设备卡号
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | rank@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V |   -   | -  |   10   |    -     |      1.5      |
| 8p-竞品V | - | - |  300   |    -     |      1.5      |
|  1p-NPU  |   -   | 237.004  |   10    |    O1    | 1.8 |
|  8p-NPU  |  82.16  | 1719.3  |  300   |    O1    | 1.8 |

# 版本说明

## 变更

2022.03.18：首次发布。

## FAQ

无。




