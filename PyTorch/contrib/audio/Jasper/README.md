# Jasper for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

Jasper语音识别网络是基于注意力机制的编码器-解码器架构，如Listen、Attend和Spell(LAS)可以将传统自动语音识别(ASR)系统上的声学、发音和语音模型组件集成到单个神经网络中。在结构上，我们证明了词块模型可以用来代替字素。我们引入了新型的多头注意力架构，它比常用的单头注意力架构有所提升。在优化方面，我们探索了同步训练，定期采样，平滑标签（label smoothing）,也应用了最小误码率优化，这些方法都提升了准确度。我们使用一个单项LSTM编码器进行串流识别并展示了结果。

- 参考实现：

  ```
  url=https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper
  commit_id=0e279a3c7cbfabacbcecb9b5f123d4b532d799f1
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

   用户可以自行获取原始数据集，可以选用开源数据集包括Librispeech等。也可以直接进入源码包根目录下运行以下两个命令下载数据集并进行预处理。脚本中的DATA_ROOT_DIR为数据集输出目录。

    ```
    bash scripts/download_librispeech.sh
    bash scripts/preprocess_librispeech.sh
    ```

   数据集目录结构参考如下所示。

   ```
   ├── dataset
         ├──dev-clean-wav
         ├──dev-other-wav
         │──librispeech-train-clean-100-wav.json
         │──librispeech-train-clean-360-wav.json      
         ├──librispeech-train-clean-500-wav.json
         │──librispeech-dev-clean-wav.json
         │──librispeech-dev-other-wav.json
         ├──librispeech-test-clean-wav.json                     
         ├──librispeech-test-other-wav.json   
         ├──test-clean-wav
         │──test-other-wav
         │──train-clean-100-wav     
         ├──train-clean-360-wav
         │──train-clean-500-wav
                  
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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/
     ```

   --data_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --amp                               //是否使用混合精度
   --data_mode                         //训练数据集名称
   --data_path                         //数据集目录   
   --eavl_data_mode                    //验证集名称
   --max_iter                          //最多迭代步数，控制训练步数，如果为-1，则不加以控制  
   --start_epoch                       //从第几个epoch开始训练
   --learning-rate                     //学习率
   --weight-decay                      //权重衰减
   --prediction-frequency              //在dev set评估之间的steps
   --resume                            //权重路径
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小
   --backend                           //后台通信方式
   --dist-url                          //设置分布式训练网址
   --distributed                       //是否使用多卡训练
   --seed                              //随机种子
   ```


# 训练结果展示

**表 2**  训练结果展示表

| 名称   | WER      | 性能/fps       | Epochs |
| :------: | :------:  | :------: | :------: |  
| GPU-1p   |    -    |   10       | 1 |     
| GPU-8p   |  10.73  |   78      | 30 |
| NPU-1p   |    -    |   4       | 1 |
| NPU-8p   |  10.89  |  34     | 30 |


# 版本说明

## 变更

2023.1.10：更新readme，重新发布。


## 已知问题

暂无。