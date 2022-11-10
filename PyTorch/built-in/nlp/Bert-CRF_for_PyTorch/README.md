# Bert-CRF for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述

## 简述

bert4torch是一个基于pytorch的训练框架，前期以效仿和实现bert4keras的主要功能为主，方便加载多类预训练模型进行finetune，提供了中文注释方便用户理解模型结构。主要是期望应对新项目时，可以直接调用不同的预训练模型直接finetune，或方便用户基于bert进行修改，快速验证自己的idea；节省在github上clone各种项目耗时耗力，且本地文件各种copy的问题。

- 参考实现：

  ```
  url=https://github.com/Tongjilibo/bert4torch
  commit_id=43c28f9dbc5fe20b9ae57fb5050658dca617f3d1
  ```
- 适配昇腾 AI 处理器的实现：

    ```
    url=https://gitee.com/ascend/ModelZoo-PyTorch.git
    code_path=PyTorch/built-in/nlp
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
  | 固件与驱动 | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```
- 本模型在X86平台上性能显著高于ARM。
- 请注意开启cpu性能模式，否则会影响模型性能，详细参考 [将cpu设置为performance模式](https://gitee.com/ascend/pytorch/blob/master/docs/zh/PyTorch%E8%AE%AD%E7%BB%83%E8%B0%83%E4%BC%98&%E5%B7%A5%E5%85%B7%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97/PyTorch%E8%AE%AD%E7%BB%83%E8%B0%83%E4%BC%98&%E5%B7%A5%E5%85%B7%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md#%E5%B0%86cpu%E8%AE%BE%E7%BD%AE%E4%B8%BAperformance%E6%A8%A1%E5%BC%8F)。
  


## 准备数据集

1. 获取数据集。

   主要参考[bert4torch](https://github.com/Tongjilibo/bert4torch)进行人民日报NER数据集准备。
   用户需自己新建一个`$data_path`路径，用于放预训练模型和数据集，`$data_path`可以设置为服务器的任意目录（注意存放的磁盘需要为NVME固态硬盘）。
   下载[人民日报数据集](https://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz)，解压到`$data_path`下。
   - `$data_path`目录结构如下：
    ```
    $data_path
    └── china-people-daily-ner-corpus
        ├── example.dev
        ├── example.test
        └── example.train
    ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

## 获取预训练模型

- 需新建`$data_path/pretrained_model`，下载[Bertbase chinese预训练模型](https://huggingface.co/bert-base-chinese/tree/main)，将下载好的文件放在`$data_path/pretrained_model`下。

- `$data_path`最终的目录结构如下：
    ```
    $data_path
    ├── china-people-daily-ner-corpus
    │   ├── example.dev
    │   ├── example.test
    │   └── example.train
    └── pretrained_model
        ├── config.json
        ├── pytorch_model.bin
        ├── tokenizer_config.json
        ├── tokenizer.json
        └── vocab.txt
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

| NAME     | Accuracy-Highest |  samples/s | AMP_Type |
| -------  | -----  | ---: | -------: |
| 1p-竞品A  | best_F1: 0.95499 | 97 |       O1 |
| 1p-NPU   | best_F1: 0.95325 | 114.333 |       O1 |
| 8p-竞品A  | best_F1: 0.92541 | 719.1 |       O1 |
| 8p-NPU   | best_F1: 0.93978 | 1040.2 |       O1 |

备注：本模型在X86平台上性能显著高于ARM，若要达到表中性能，请注意必须在x86平台上进行性能测试，并将cpu设置为performance模式。

# 版本说明

## 变更

2022.10.12：首次发布

## 已知问题


无。
