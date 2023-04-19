# CRNN for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

CRNN (Convolutional Recurrent Neural Network) 于2015年由华中科技大学的白翔老师团队提出，直至今日，仍旧是文本识别领域最常用也最有效的方法。

- 参考实现：

  ```
  url=https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec.git
  commit_id=90c83db3f06d364c4abd115825868641b95f6181
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification
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

   模型训练以 MJSynth 数据集为训练集，IIIT 数据集为测试集。

   用户自行下载并解压 data_lmdb_release.zip，将其中的data_lmdb_release/training/MJ/MJ_train 文件夹 (重命名为 MJ_LMDB) 和 
   data_lmdb_release/evaluation/IIIT5k_3000 文件夹 (重命名为 IIIT5k_lmdb)上传至服务器的任意目录下，作为数据集目录。
   > 注意：若用户选择下载原始数据集，则需要将其转换为 lmdb 格式数据集，再根据上述步骤进行数据集上传。

   数据集目录结构参考如下所示：
   ```
   ├──服务器任意目录下
       ├──MJ_LMDB
             │──data.mdb
             │──lock.mdb
       ├──IIIT5K_lmdb
             │──data.mdb
             │──lock.mdb
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
     bash ./test/train_performance_1p.sh --data_path=数据集路径    # 单卡性能
     
     bash ./test/train_full_1p.sh --data_path=数据集路径           # 单卡精度 
     ```

   - 单机8卡训练

     启动8卡训练。
     ```
     bash ./test/train_performance_8p.sh --data_path=数据集路径    # 8卡性能
     
     bash ./test/train_full_8p.sh --data_path=数据集路径           # 8卡精度
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                         //训练集路径
   --epochs                            //重复训练次数
   --npu                               //npu训练卡id设置
   --max_step                          //设置最大迭代次数
   --stop_step                         //设置停止的迭代次数
   --profiling                         //设置profiling的方式
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|  NAME  | Acc@1 |    FPS    | Epochs | AMP_Type | Torch_Version |
| :----: | :---: | :-------: | :----: | :------: | :-----------: |
| 1p-NPU |   -   | 11733.53  |   1    |    O2    |      1.8      |
| 8p-NPU | 0.75  | 106510.27 |  100   |    O2    |      1.8      |


# 版本说明
2022.02.17：更新readme，重新发布。

## FAQ
无。
