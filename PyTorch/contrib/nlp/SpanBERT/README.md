# SpanBERT for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

SpanBERT在BERT的基础上，采用Geometric Spans的遮盖方案并加入Span Boundary Objective (SBO) 训练目标，通过使用分词边界的表示以及被遮盖词元的位置向量来预测被遮盖的分词的内容，增强了 BERT 的性能，特别是在一些与 Span 相关的任务，如抽取式问答。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/SpanBERT.git
  commit_id=96f2dfbede280df3a5d146425a9c8eca7b425d41
  ```

- 适配昇腾 AI 处理器的实现：

    ```
    url=https://gitee.com/ascend/ModelZoo-PyTorch.git
    code_path=PyTorch/contrib/nlp
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

1. 获取数据集。

   请用户需自行获取SQuAD 1.1数据集，下载地址为:[https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset](https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset)。上传数据集到服务器任意目录并解压。

   数据集目录结构如下所示：  

       ├── SQuAD 1.1
       	├──train-v1.1.json                 
       	├──dev-v1.1.json  

   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。

## 获取预训练相关文件

1. 下载dict.txt文件。

   请用户需自行获取，下载地址为[https://github.com/facebookresearch/SpanBERT/tree/main/pretraining]( https://github.com/facebookresearch/SpanBERT/tree/main/pretraining)，将其放在SpanBERT/pretraining目录下。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡性能和单机8卡训练。

   - 单机单卡性能

     启动单卡性能。

     ```
     bash ./test/train_performance_1p.sh --data_path=数据集路径
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=数据集路径 
     ```

    --data_path参数填写数据集路径。

    模型训练脚本参数说明如下。  
   ```
   公共参数：
    --data_path                           //数据集路径     
    --train_epochs                        //重复训练次数
    --batch_size                          //训练批次大小
    --learning_rate                       //初始学习率
    --RANK_ID                             //默认卡号
    --ASCEND_DEVICE_ID                    //默认设备号
    --test_path_dir                       //包含test文件夹的路径
   ```
   
   训练完成后，权重文件默认会写入到squad_output目录下，并输出模型训练精度和性能信息到test下output文件夹内。


# 训练结果展示

**表 2**  训练结果展示表

|  F1  | FPS  | Npu_nums | Epochs | AMP_Type | loss scale | Torch |
| :--: | :--: | :------: | :----: | :------: | :--------: | :---: |
|  -   | 24.3 |    1     |   4    |    O2    |   128.0    |  1.5  |
| 91.9 | 47.2 |    8     |   4    |    O2    |   128.0    |  1.5  |
|  -   | 13.9 |    1     |   1    |    O2    |   128.0    |  1.8  |
| 91.9 | 44.4 |    8     |   4    |    O2    |   128.0    |  1.8  |


# 版本说明

## 变更

2022.11.01：更新torch1.8版本，重新发布。

2022.07.01：首次发布。

## 已知问题

当网络环境不能访问`https://dl.fbaipublicfiles.com`时，删除SpanBERT/code/pytorch_pretrained_bert目录下的file_utils.py，并将file_utils_for_network.py文件名修改为file_utils.py，然后在SpanBERT/cache目录下下载好预训练模型。

网址:

 'spanbert-base-cased': `https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf_base.tar.gz`,    

 'spanbert-large-cased': `https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf.tar.gz`

