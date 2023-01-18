# DeepSpeech for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

DeepSpeech2是一个建立在端到端深度学习之上，将大多数模块替换为单个模型的第二代ASR语音系统。其ASR管道在几个基准上的精确度接近甚至超过了Amazon人工的精度，可以再多种语言下工作，并且可以部署在生产环境中。

- 参考实现：

  ```
  url=https://github.com/SeanNaren/deepspeech.pytorch
  commit_id=b00d17387ca47b05b8a3c0ccc91a133eb4966b40  
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
  | 硬件 | [1.0.11.SPC002](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [21.0.2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.0.2](https://www.hiascend.com/software/cann/commercial?version=5.0.2)    |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/)                      |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 编译安装wrap-ctc模块。

    ```shell
    ### npu环境变量
    source {deepspeech_root}/test/env_npu.sh
    git clone https://github.com/SeanNaren/warp-ctc.git
    cd warp-ctc
    git checkout -b pytorch_bindings origin/pytorch_bindings
    mkdir build; cd build; cmake ..; make
    cd ../pytorch_binding && python3.7 setup.py install
    ```

- 安装依赖。

  ```
  pip3.7 install -r requirements.txt
  ```
  
- 如果需要在多机或者多卡上训练该模型，那么需要按照以下步骤安装etcd。

    ```shell
    sudo apt-get install etcd
    sudo apt-get install sox
    ```


## 准备数据集

1. 获取数据集。

   ```shell
   cd data
   python3.7 an4.py
   ```

2. 或者您还可以自行下载数据集解压至源码包根目录下的`data/`文件夹下。

    执行完成后目录结构如下所示：
   ```
   ├── data
      ├──an4_train_manifest.csv
      ├──an4_val_manifest.csv 
      ├──an4_test_manifest.csv  
      ├──an4_dataset
         ├──train                    
         ├──val
         ├──test          
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
     bash ./test/train_full_1p.sh --data_path=./data/         # 精度训练
     bash ./test/train_performance_1p.sh --data_path=./data/  # 性能训练
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=./data/           # 精度训练
     bash ./test/train_performance_8p.sh --data_path=./data/    # 性能训练   
     ```

   模型训练脚本参数说明如下。

   ```
   --data.num_workers                           //加载数据进程数      
   --training.epochs                             //重复训练次数
   --data.batch_size                             //训练批次大小，默认：240
   --optim.learning_rate                         //初始学习率，默认：1
   --apex.loss_scale                             //混合精度lossscale大小
   ```

# 训练结果展示

**表 2**  训练结果展示表

| NAME  |  WER   |  CER   | Epochs | APEX | FPS  |
| :---: | :----: | :----: | :----: | :--: | :--: |
| NPU1P | 9.444  | 5.723  |   70   |  O2  |  4   |
| NPU8P | 17.464 | 10.926 |   70   |  O2  |  22  |
| GPU1P | 10.349 | 7.076  |   70   |  O2  |  94  |
| GPU8P | 15.265 | 9.834  |   70   |  O2  | 377  |


# 版本说明

## 变更

2022.12.20：整改Readme，重新发布。

## 已知问题

无。

