# Tacotron2 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

Tacotron2是一个从文字直接转化为语音的神经网络。这个体系是由字符嵌入到梅尔频谱图的循环序列到序列神经网络组成的，然后是经过一个修改过后的WaveNet，该模型的作用是将频谱图合成波形图。
- 参考实现：

  ```
  url=https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2/
  commit_id=9a6c5241d76de232bc221825f958284dc84e6e35  
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

  | 配套       | 版本                                                                          |
  |-----------------------------------------------------------------------------| ------------------------------------------------------------ |
  | 硬件       | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/)                      |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  

- 安装依赖。

  ```
  pip3.7 install -r requirements.txt
  ```
  LLVM版本与numbra、llvmlite版本号严格依赖，如LLVM 7.0对应llvmlite的0.30.0，numbra的0.46.0版本。



## 准备数据集

1. 获取数据集。

   用户自行下载LJSpeech-1.1数据集，解压并置于模型脚本根目录下，然后在模型脚本根目录下运行scripts/prepare_mels.sh。

    ```
    wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
    bash scripts/prepare_mels.sh    
    ```

    初次预处理时间较长，请耐心等待。

    数据集目录结构参考如下所示。

    ```
    ├──LJSpeech-1.1
        ├── mels            
        ├── metadata.csv            
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

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=./LJSpeech-1.1

     bash ./test/train_performance_1p.sh --data_path=./LJSpeech-1.1
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=./LJSpeech-1.1

     bash ./test/train_performance_8p.sh --data_path=./LJSpeech-1.1
     ```

   模型训练脚本参数说明如下。

   ```
   公共参数：
   -m                                  //训练模型名称
   -0                                  //训练文件输出路径  
   --amp                               //是否使用apex混合精度训练
   --train_epochs                      //重复训练次数
   --bs                                //训练批次大小
   --lr                                //初始学习率
   --seed                              //随机种子
   ```


# 训练结果展示

**表 2**  训练结果展示表

| NAME | Accuracy |    FPS    | Npu_nums | Epochs   | AMP_Type |
| :------: | :------: | :------:  | :------: | :------: | :------: |
| GPU |     |      | 1        | 1        | O2       |
| GPU |     |      | 8        | 301      | O2       |
| NPU |     |      | 1        | 1        | O2       |
| NPU |     |      | 8        | 301      | O2       |


# 版本说明

## 变更

2023.1.12：整改Readme，重新发布。

## 已知问题

无。

