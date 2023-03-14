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


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |

- 环境准备指导。

   1. 请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》准备torch_npu环境。
   2. 安装torchaudio，npu安装方法请参考:https://gitee.com/ascend/modelzoo/issues/I48AZM
  
- 安装依赖并安装环境。

    ```
    pip install -r requirements.txt
    pip install --editable .
    ```

```
注意：安装依赖环境时，如果自动卸载torch，可以将'pip install --editable .'替换为'pip install --editable . --no-deps'。
```


## 准备数据集

1. 获取数据集。

    用户自行获取 `train-clean-5` 数据集和 `rirs_noises` 数据集，将数据集分别解压至服务器任意目录下新建的 `data/LibriSpeech` 和 `data/RIRS_NOISES` 文件夹路径下，数据集目录结构参考如下：

   ```
   ├── data
       ├──LibriSpeech
          ├──train-clean-5
                                                
       ├──RIRS_NOISES
          ├──pointsource_noises
          ├──real_rirs_isotropic_noises
          ├──simulated_rirs           
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

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
     bash ./test/train_full_1p.sh --data_folder=/data/xxx/   # 单卡精度
     bash ./test/train_performance_1p.sh --data_folder=/data/xxx/  # 单卡性能 
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_folder=/data/xxx/   # 8卡精度
     bash ./test/train_performance_8p.sh --data_folder=/data/xxx/  # 8卡性能    
     ```

   --data_folder参数填写数据集路径，需写到数据集的一级目录。
   
3. 日志文件夹如下。

     ```
     Log path:
        test/output/train_full_1p.log              # 单卡精度日志
        test/output/train_performance_1p.log       # 单卡性能日志
        test/output/train_full_8p.log              # 8卡精度日志
        test/output/train_performance_8p.log       # 8卡性能日志
     ```

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_folder                       //数据集路径
   --local_rank                        //训练设备ID
   --batch_size                        //训练批次大小
   --number_of_epochs                  //训练重复次数
   多卡训练参数：
   --distributed_launch                //开启多卡训练
   --distributed_backend               //多卡通信协议
   ```

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Valid Err |   FPS | Epochs | AMP_Type | Torch_Version |
|:-------:|:---------:|:-----:|:------:|:--------:| :---: |
| 1p-竞品V  | -         | 17.43 | 1      |        - | 1.5 |
| 8p-竞品V  | 7.81e-03  | 83.26 | 5      |        - | 1.5 |
| 1p-NPU  | -         |  9.548 | 1      |       O1 | 1.8 |
| 8p-NPU  | 3.91e-02  | 73.801 | 5      |       O1 | 1.8 |

# 版本说明

## 变更

2022.07.05：整改Readme，重新发布。

## FAQ

无。
