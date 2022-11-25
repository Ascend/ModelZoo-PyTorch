# Wav2Vec2.0

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述

## 简述

Wav2vec2.0是Meta在2020年发表的无监督语音预训练模型。它的核心思想是通过向量量化（Vector Quantization，VQ）构造自建监督训练目标，对输入做大量掩码后利用对比学习损失函数进行训练。

- 参考实现：
  
  ```
  url=https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec
  branch=master
  commit_id=a0ceabc287e26f64517fadb13a54c83b71e8e469
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

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动 | [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip3 install -r requirements.txt
  apt-get install libsndfile1 (yum installl libsndfile1)
  pip3 uninstall fairseq
  pip3 install -e ./
  ```
  


## 准备数据集

1. 获取数据集。

   主要参考[wav2vec2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)进行LibriSpeech数据集准备。
   用户需自己新建一个`$data_path`路径，用于放预训练模型和数据集，`$data_path`可以设置为服务器的任意目录（注意存放的磁盘需要为NVME固态硬盘）。
   下载LibirSpeed数据集，包括train-clean-100，dev-clean，按照[wav2vec2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)准备manifest，统一放置到`$data_path`下。
   - `$data_path`目录结构如下：
    ```
    $data_path
    ├── train-clean-100
    ├── dev-clean
    └── manifest

    ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

## 获取预训练模型

- 需下载[Wav2vec2.0预训练模型](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt)，将下载好的文件放在`$data_path`下。

- `$data_path`最终的目录结构如下：
    ```
    $data_path
    ├── train-clean-100
    ├── dev-clean
    ├── wav2vec_small.pt
    └── manifest

    ```

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机8卡，单机单卡。

   - 单机单卡训练

     启动单卡训练

     ```
     bash ./test/train_full_1p.sh --data_path=$data_path
     ```
     ```
     bash ./test/train_performance_1p.sh --data_path=$data_path
     ```
    
     训练完成后，输出模型训练精度和性能信息。

   - 单机8卡训练

     启动8卡训练

     ```
     bash ./test/train_full_8p.sh --data_path=$data_path
     ```
     ```
     bash ./test/train_performance_8p.sh --data_path=$data_path
     ```
     `--data_path`参数填写数据集根目录。

   - 模型训练脚本参数说明如下。

      ```
      公共参数：
      --train_epochs                      //训练的总epochs数
      --workers                           //dataloader开启的线程数
      ```
    
     训练完成后，权重文件默认会写入到和test文件同一目录下，并输出模型训练精度和性能信息到网络脚本test下output文件夹内。


# 训练结果展示

**表 2**  训练结果展示表

|  名称  | wer  | 性能 |
| :----: | :---: | :--: |
| GPU-1p |   -   | 5524.7  |
| GPU-8p | 5.443  | 44493.3 |
| NPU-1p |   -   | 4869.8  |
| NPU-8p | 5.546 | 33463.9 |
| 实测1.5-1p |   -   | 5526.3  |
| 实测1.5-8p | 5.57 | 33474.3 |


# 版本说明

## 变更

2022.11.24：首次发布

## 已知问题


无。