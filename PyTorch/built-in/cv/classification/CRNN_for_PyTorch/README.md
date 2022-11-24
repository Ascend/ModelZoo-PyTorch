# CRNN_for_PyTorch

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

  | 配套 | 版本                                                                           |
  |------------------------------------------------------------------------------| --- |
  | 硬件 | [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)  |
  | NPU固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)                       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。
  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   模型训练以 MJSynth 数据集为训练集，IIIT 数据集为测试集。

   用户需点击 [链接](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0) 下载并解压 data_lmdb_release.zip，
   将其中的 data_lmdb_release/training/MJ/MJ_train 文件夹 (重命名为 MJ_LMDB) 和 
   data_lmdb_release/evaluation/IIIT5k_3000 文件夹 (重命名为 IIIT5k_lmdb)
   上传至服务器的任意目录下，作为数据集目录。

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
     bash ./test/train_performance_1p.sh --data_path=数据集路径    # 1p性能
     bash ./test/train_full_1p.sh --data_path=数据集路径           # 1p精度 
     ```

   - 单机8卡训练

     启动8卡训练。
     ```
     bash ./test/train_performance_8p.sh --data_path=数据集路径    # 8p性能
     bash ./test/train_full_8p.sh --data_path=数据集路径           # 8p精度
     ```

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                         //训练集路径
   --epochs                            //训练周期
   ```
   

# 训练结果展示

**表 2**  训练结果展示表

| Acc@1 |    FPS    | Npu_nums | Epochs | AMP_Type | Torch |
|:-----:|:---------:|:--------:|:------:|:--------:|:-----:|
|   -   |   10455   |    1     |   1    |    O2    |  1.5  |
| 76.80 |   34308   |    8     |  100   |    O2    |  1.5  |
|   -   | 10834.792 |    1     |   1    |    O2    |  1.8  |
| 75.33 | 84898.343 |    8     |  100   |    O2    |  1.8  |


# 版本说明
2022.08.01：更新pytorch1.8版本，重新发布。

## 已知问题
无。
