# WaveGlow for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

WaveGlow是一种基于流的网络，能够从梅尔谱图生成高质量的语音。WaveGlow结合了WaveNet的实现，以提供快速、高效和高质量的音频合成，无需自动回归。WaveGlow仅使用单个网络实现，仅使用单个loss函数进行训练，最大化训练数据的似然，这使得训练过程简单稳定。

- 参考实现：

  ```
  url=https://github.com/NVIDIA/waveglow.git
  commit_id=8afb643df59265016af6bd255c7516309d675168
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

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  在模型源码包根目录下执行命令。
  ```
  pip3.7 install -r requirements.txt

  # 检查安装依赖是否成功
  import librosa

  # 若import librosa提示 sndfile 找不到
  apt-get/yum install libsndfile

  # 若 scikit-learn 报错，则使用conda安装
  pip3.7 install scikit-learn

  # 若numpy被前面安装的内容更新，则重新安装(numpy版本需1.20及以下)
  pip3.7 install numpy==1.20

  # 若 sympy 报错，则pip/conda安装
  pip3.7 install sympy
  ```

## 准备数据集


1. 获取数据集。

   用户可以自行获取原始数据集，可以选用开源数据集包括LJ-Speech等。将数据集上传到源码包根目录下新建的“data/”文件夹下并解压。

   以LJ-Speech数据集为例，数据集目录结构参考如下所示。

   ```
   ├── data
         ├──wavs
             ├──LJ001-001.wav
             │──LJ001-002.wav
             │──LJ001-XXX.wav     
   ```
   
   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。

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
     
     bash ./test/train_performance_1p.sh --data_path=数据集路径 --output_directory=./checkpoints  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=数据集路径 --output_directory=./checkpoints  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=数据集路径 --output_directory=./checkpoints  # 8卡性能
     ```
     
   - 单机8卡评测
     
     启动8卡评测。
     
     ```
     bash ./test/train_eval_8p.sh --data_path=数据集路径 --pth_path=./checkpoints/waveglow_21000  # 8卡评测     
     ```
   
   --data_path参数填写数据集路径，需写到数据集的一级目录。
   
   --pth_path参数填写训练权重生成路径，需写到权重文件的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                         //数据集目录   
   --output_directory                  //训练输出pth模型文件地址
   --pth_path                          //推理使用pth文件地址
   --checkpoint_path                   //迁移学习采用pth文件地址  
   --fp16_run                          //是否使用apex混合精度训练
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小
   --learning_rate                     //初始学习率
   --sigma                             //sigma函数初始值
   --iters_per_checkpoint              //每间隔多少iter保存一下pth文件
   --c                                 //glow网络配置json文件
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表


| NAME | Accuracy |    FPS    | Epochs   | AMP_Type | Torch_Version |
| :------: | :------: | :------:  | :------: | :------: | :------: |
| 1p-竞品V | -        | 0.007     | 1        | O2       | 1.5 |
| 8p-竞品V | -5.6     | 0.42      | 313      | O2       | 1.5 |
| 1p-NPU | -        | 0.002     | 1        | O2       | 1.8 |
| 8p-NPU | -5.6     | 0.21      | 313      | O2       | 1.8 |



# 版本说明

## 变更

2023.1.10：更新readme，重新发布。


## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md