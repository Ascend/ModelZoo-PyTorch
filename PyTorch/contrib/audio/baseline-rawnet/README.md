# Baseline-Rawnet for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果](训练结果.md)
-   [版本说明](版本说明.md)

# 概述

## 简述

最近，使用深度神经网络对原始波形进行直接建模已被广泛研究用于音频领域的许多任务。然而，在说话人验证中，原始波形的利用处于初步阶段，需要进一步研究。在这项研究中，我们探索了输入原始波形以改进各个方面的端到端深度神经网络：前端说话人嵌入提取，包括模型架构、预训练方案、附加目标函数和后端分类。使用预训练方案调整模型架构可以提取说话人嵌入，从而显着提高性能。

- 参考实现：

  ```
  url=https://github.com/Jungjee/RawNet
  commit_id=585ce2c4fb80ae6ab236f79f06911e2f8bef180c
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/audio
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
  | 硬件       | [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```
  pip install -r requirements.txt
  ```

## 准备数据集

1. 准备数据集

   请用户自行准备好数据集，包含训练集和验证集两部分，可选用的数据集包括 [VoxCeleb2，VoxCeleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) 等。在源码包根目录下建立“DB”/文件夹，将准备好的数据集上传至“DB/”文件夹中并解压。数据集很大。下载和解压缩时，请确保有足够的硬盘空间。

   解压后评估数据集和训练数据集分别位于“DB/VoxCeleb1”和“DB/VoxCeleb2”文件夹路径下，该目录下每个文件夹代表一个类别，且同一文件夹下的所有图片都有相同的标签。当前提供的训练脚本中，是以VoxCeleb1和VoxCeleb2数据集为例。在使用其他数据集时，修改数据集路径。

2. 数据预处理

   从上面的url下载的VoxCeleb2中的数据格式为.m4a。如果不使用已转换的数据集，则应首先执行数据预处理脚本，将数据格式转为.wav。

   ```
   python m4a2wav.py
   ```

   数据集目录结构参考：

   ```
   ${RawNet}/DB/VoxCeleb1/
   ├── dev_wav
   │   ├── id10001
   │   │   ├── 1zcIwhmdeo4
   │   │   │   ├── 00001.wav
   │   │   │   ├── 00002.wav
   │   │   │   └── 00003.wav
   │   │   ├── 5ssVY9a5X-M
   │   │   │   ├── 00001.wav
   │   │   │   ├── 00002.wav
   │   │   │   ├── 00003.wav
   │   │   │   └── 00003.wav
   │   └── ...
   ├── eval_wav
   │   ├── id10270
   │   │   ├── 5r0dWxy17C8
   │   │   │   ├── 00001.wav
   │   │   │   ├── 00002.wav
   │   │   │   ├── 00003.wav
   │   │   │   ├── 00004.wav
   │   │   │   └── 00005.wav
   │   └── ...
   │       ├── _z_BR0ERa9g
   │           ├── 00001.wav
   │           ├── 00002.wav
   │           └── 00003.wav
   ├── val_trial.txt
   └── veri_test.txt 
   
   ${RawNet}/DB/VoxCeleb2/
   └── wav
       ├── id00012
       │   ├── 21Uxsk56VDQ
       │   │   ├── 00001.wav
       │   │   ├── ...
       │   │   └── 00059.wav
       │   ├── 00-qODbtozw
       │   │   ├── ...
       │   │   ├── 00079.wav
       │   │   └── 00080.wav
       ├── ...
       │   └── zw-4DTjqIA0
       │       ├── 00108.wav
       │       └── 00109.wav
       └── id09272
           └── u7VNkYraCw0
               ├── ...
               └── 00027.wav
   ```

# 开始训练

1. 进入运行脚本目录下

   ```
   cd /${模型文件夹名称}/test
   ```

2. 运行训练脚本

   该模型支持单机单卡训练和单机8卡训练。

   * 单机单卡训练

     启动单卡训练

     ```
     # 1p train perf
     bash train_performance_1p.sh --data_path=./DB
     # 1p train full
     bash train_full_1p.sh --data_path=./DB
     ```

     * 参数说明：

       * data_path：填写数据集路径

       * 代码运行日志保存在**test**路径下的**output**文件夹

   * 单机多卡训练

     启动8卡训练

     ```
     # 8p train perf
     bash train_performance_8p.sh --data_path=./DB
     # 8p train full
     bash train_full_8p.sh --data_path=./DB
     ```

     * 参数说明：

       * data_path：填写数据集路径

       * 代码运行日志保存在**test**路径下的**output**文件夹

   遵循“output”的目录结构如下：

   ```
   ${RawNet}/train/train_${device_count}P
   |-- DNNS/${name}/
   |   |-- models
   |   |   |--best_opt_eval.pt ## The best perfomance model is saved here
   |   |   |--TA_${epoch}_${eer}.pt ##The other model is saved here
   |   |-- results
   |   |-- log
   |   |   |-- eval_epoch${epoch}.txt   ## The training log is saved here
   |   |-- prof
   |   |-- eers.txt  ##The eers is saved here
   |   |-- f_params.txt ##The params of the model are saved here
   ```

# 训练结果展示

**表 2**  训练结果展示表

|       eer(percentage)       | FPS(aver) | Npu_nums | Epochs | AMP_Type | Torch |
| :-------------------------: | :-------: | :------: | :----: | :------: | ----- |
|            0.14             |   7760    |    1     |   1    |    O2    | 1.5   |
|              -              |   7912    |    1     |   1    |    O2    | 1.8   |
| 0.038(aver) and 0.035(high) |   8573    |    8     |   20   |    O2    | 1.5   |
| 0.038(aver) and 0.035(high) |   12575   |    8     |   20   |    O2    | 1.8   |



# 版本说明

## 变更

2022.03.18：首次发布

2022.11.24：更新pytorch1.8版本，重新发布。

## 已知问题

无。

# 公网地址说明

代码涉及公网地址参考 ```./public_address_statement.md```
