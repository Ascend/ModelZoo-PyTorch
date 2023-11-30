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


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.2.2.post3；pillow==8.4.0 |
  | PyTorch 1.8 | torchvision==0.9.1；pillow==8.4.0 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。


## 准备数据集

1. 获取数据集。

   请用户需自行获取SQuAD 1.1数据集，将下载好的数据集上传到服务器任意目录并解压。
   数据集目录结构如下所示：

       ├── SQuAD 1.1
             ├──train-v1.1.json                 
             ├──dev-v1.1.json  

   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。

## 获取预训练相关文件

1. 下载dict.txt文件。

   请用户根据“参考实现”源码链接，将源码 `SpanBERT/pretraining` 目录下的 dict.txt 文件下载到本模型 `SpanBERT/pretraining`  目录下。


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
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

    模型训练脚本参数说明如下。  
   ```
   公共参数：
   --train_file                          //数据集路径
   --batch_size                          //训练批次大小
   --npu_id                              //设置训练卡id
   --learning_rate                       //初始学习率
   --loss_scale                          //loss scale大小
   --num_train_epochs                    //训练周期数
   --do_train                            //设置是否进行训练
   --seed                                //随机数种子设置
   ```
   
   训练完成后，权重文件默认会写入到squad_output目录下，并输出模型训练精度和性能信息到test下output文件夹内。


# 训练结果展示

**表 2**  训练结果展示表

|  NAME  |  F1  | FPS  | Epochs | AMP_Type | Torch_Version |
| :----: | :--: | :--: | :----: | :------: | :-----------: |
| 1p-NPU |  -   | 13.9 |   1    |    O2    |      1.8      |
| 8p-NPU | 91.9 | 44.4 |   4    |    O2    |      1.8      |


# 版本说明

## 变更

2023.03.13：更新readme，重新发布。

2022.07.01：首次发布。

## FAQ

1. 当网络环境不能访问`https://dl.fbaipublicfiles.com`时，删除SpanBERT/code/pytorch_pretrained_bert目录下的file_utils.py，将file_utils_for_network.py文件名修改为file_utils.py，根据实际需要在SpanBERT/cache目录下下载好预训练模型。

   ```
   spanbert-base-cased:  https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf_base.tar.gz
   spanbert-large-cased: https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf.tar.gz
   ```

   
# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
