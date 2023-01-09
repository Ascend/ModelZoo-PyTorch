#  TDNN for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述

## 简述
ECAPA-TDNN是一个用于声纹识别的深度学习模型，它基于传统TDNN模型进行了改进，主要有三个方面的优化，分别是：增加了一维SE残差模块（1-Dimensional Squeeze-Excitation Res2Block）;多层特征融合（Multi-layer feature aggregation and summation）;通道和上下文相关的统计池化（Channel- and context-dependent statistics pooling）。

- 参考实现：

  ```
  url=https://github.com/speechbrain/speechbrain/tree/develop/templates/speaker_id
  commit_id=d333cf277706146bd622cb46f928083f9938b21a
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/audio
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}        # 克隆仓库的代码 
  cd {code_path}         # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                                                 |
  |------------------------------------------------------------------------------------| ------------------------------------------------------------ |
  | 硬件 | [1.0.11.SPC002](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [21.0.2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)        |
  | CANN       | [5.0.2](https://www.hiascend.com/software/cann/commercial?version=5.0.2)           |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/)                             |

1. 环境准备指导。

   1. 请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》准备torch_npu环境。
   2. 安装torchaudio，npu安装方法请参考:https://gitee.com/ascend/modelzoo/issues/I48AZM
  
4. 安装依赖并安装环境。

    ```
    pip install -r requirements.txt
    pip install --editable .
    ```


## 准备数据集

1. 获取数据集。

   模型训练使用rirs_noises数据集，数据集请用户自行获取。

2. 将数据集分别解压至`./data/LibriSpeech`和`./data/RIRS_NOISES`文件夹路径下，数据集目录结构参考：

   ```
   ├── data
   │    ├──LibriSpeech├──train-clean-5
   │    │                                         
   │    ├──RIRS_NOISES├──pointsource_noises
                      ├──real_rirs_isotropic_noises
                      ├──simulated_rirs           
   ```



# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}/templates/speaker_id
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_folder=/data/xxx/   # 精度训练
     bash ./test/train_performance_1p.sh --data_folder=/data/xxx/  # 性能训练 
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_folder=/data/xxx/   # 精度训练
     bash ./test/train_performance_8p.sh --data_folder=/data/xxx/  # 性能训练    
     ```

    --data_folder参数填写数据集路径。
   
3. 日志文件夹如下。

     ```
     Log path:
        test/output/train_full_1p.log              # 1p training result log
        test/output/train_performance_1p.log       # 1p training performance result log
        test/output/train_full_8p.log              # 8p training result log
        test/output/train_performance_8p.log       # 8p training performance result log
     ```

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_folder                              //数据集路径
   --workers                           //加载数据进程数
   ```

# 训练结果展示

**表 2**  训练结果展示表


| NAME    | Valid Err |   FPS | Epochs | AMP_Type |
|---------|-----------|------:|--------|---------:|
| 1p-GPU  | -         | 17.43 | 1      |        - |
| 1p-NPU  | -         |  7.99 | 1      |       O1 |
| 8p-GPU  | 7.81e-03  | 83.26 | 5      |        - |
| 8p-NPU  | 3.91e-02  | 45.78 | 5      |       O1 |

# 版本说明

## 变更

2022.07.05：整改Readme，重新发布。

## 已知问题

无。
