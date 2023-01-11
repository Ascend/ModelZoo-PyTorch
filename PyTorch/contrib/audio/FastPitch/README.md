# FastPitch for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

  FastPitch模型由双向Transformer主干(也称为Transformer编码器)，音调预测器和持续时间预测器组成。在通过第一组N个Transformer块编码后，信号用基音信息增强并离散上采样，然后它通过另一组Transformer块，目的是平滑上采样信号，并构建梅尔谱图。

- 参考实现：
    ```
    url=https://github.com/NVIDIA/DeepLearningExamples.git
    branch=master
    commit_id=8a1661b6e22416194197b8842738ad7b98e96974
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
  | 硬件       | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```
  pip3.7 install -r requirements.txt
  ```

## 准备数据集


1. 获取数据集。

   用户可以在源码包根目录下运行以下脚本自行下载LJSpeech-1.1数据集。
      ```
      bash scripts/download_dataset.sh
      bash scripts/prepare_dataset.sh   

      ```

    数据集目录结构参考如下所示。

    ```
    ./LJSpeech-1.1
    ├── mels            
    ├── metadata.csv    
    ├── pitch           
    ├── README
    └── wavs           
    ```


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练，训练之前请在训练脚本中修改默认的数据集路径，例如：./test/train_performance_1p.sh 脚本中的DATASET_PATH对应的路径修改为实际数据集所在路径。

     ```
     bash ./test/train_full_1p.sh 
     
     bash ./test/train_performance_1p.sh 
     ```

   - 单机8卡训练

     启动8卡训练，训练之前请在训练脚本中修改默认的数据集路径，例如：./test/train_performance_8p.sh 脚本中的DATASET_PATH对应的路径修改为实际数据集所在路径。

     ```
     bash ./test/train_full_8p.sh 
     
     bash ./test/train_performance_8p.sh 
     ```


3. 模型训练脚本参数说明如下。

    ```
    公共参数：
    --amp                               //是否使用混合精度
    --datasaet_path                     //数据集路径
    --lr                                //初始学习率
    --epochs                            //重复训练次数
    --batch-size                        //训练批次大小
    --num_gpus                          //使用卡数
    --weight-decay                      //权重衰减
    --epochs-per-checkpoint             //每训练N轮保存一下模型权重
    --grad-accumulation                 //训练过程中，每N个step打印一下精度及性能
    --output_dir                        //训练过程保存的模型权重路径
    ```

# 训练结果展示

**表 2**  训练结果展示表

| 名称 | Val Loss | FPS      | Npu_nums | Epochs | AMP_Type |
| ---- | -------- | -------- | -------- | ------ | -------- |
| NPU  | -        | 2084.35  | 1        | 1      | O1       |
| NPU  | 3.69     | 18736.45 | 8        | 100    | O1       |
| GPU  | -        | 7112.68  | 1        | 1      | O1       |
| GPU  | 3.49     | 49083.58 | 8        | 100    | O1       |

# 版本说明

## 变更

2023.1.10：更新readme，重新发布。


## 已知问题

暂无。