# Bert_text_classification for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

BERT的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的Encoder，是一种用于自然语言处理（NLP）的预训练技术。Bert-base模型是一个12层，768维，12个自注意头（self attention head）,110M参数的神经网络结构，它的整体框架是由多层transformer的编码器堆叠而成的。该模型完成的是文本分类的下游任务，主要针对CoLA、SST-2、MRPC、STS-B、QQP、MNLI、QNLI、RTE和WNLI这九个数据集进行评估。

- 参考实现：

  ```
  url=https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification
  commit_id=d1d3ac94033b6ea1702b203dcd74beab68d42d83
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/nlp/
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 Pytorch 版本和已知已知三方库依赖如下所示。

  **表 1**  版本支持表
 
  | Torch_Version |                        三方库依赖版本                                                          |
  |:-------------:|:-------------------------------------------------------------------------------------------:|
  |  Pytorch_1.8  | python-crfsuite==0.9.6; six==1.12.0; sklearn-crfsuite==0.3.6; tabulate==0.8.3; tqdm==4.31.1 |


- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  

- 安装依赖：

  ```
  pip install -r requirements.txt
  ```

- 安装transformers：

  ```
  cd transformers
  pip3 install -e ./
  cd ..
  ```

## 准备数据集

该模型数据集由脚本自动下载，无需手动下载。数据目录结构如下：
   ```
    $data_path
     └── test
     └── validation
     └── train
   ```
  > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。 

## 获取预训练模型
   请参考原始仓库上的README.md进行预训练模型获取。将获取的预训练模型bert-large-cased放在源码根目录下。在获取预训练模型之前需执行以下命令。
   ```
    git lfs install 
   ```
   在预训练模型下载完毕之后执行以下命令：
   ```
    cp -f ./config.json ./bert-large-cased/
   ```

# 开始训练

## 训练模型
1. 进入解压后的源码包根目录
    ```
     cd /${模型文件名称} 
     ```
2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --train_epochs=$train_epochs --TASK=$TASK  # 单卡精度性能训练 
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --train_epochs=$train_epoch --TASK=$TASK   # 8卡精度、性能训练
     bash ./test/train_performance_8p_wnli.sh               #wnli 8卡性能
     ```
    `--train_epochs`参数填写训练的总epoch数;

    `--TASK`参数填写任务的名称（从cola、sst2、mrpc、stsb、qqp、mnli、qnli、rte和wnli中选择一个填写）。
    - 模型训练脚本参数说明如下。

      ```
      公共参数：
      --dataloader_num_workers             //dataloader开启的线程数
      --do_train                          //开启训练
      --device                            //训练所使用的设备
      --do_eval                           //开启评估
      --per_device_train_batch_size       //batchsize
      --learning_rate                     //学习率参数
      --optim                             //使用的优化器
      --output_dir                        //checkpoint保存的路径
      ```
 
# 训练结果展示

**表 2**  单卡训练结果展示表

| TASK  |           Metric           | 1p-精度(竞品A) | 1p-精度(NPU)  | 1p-性能(竞品A)<br/>sample/s | 1p-性能(NPU)<br/>sample/s | AMP_Type | Epoch | Torch_Version |
|:-----:|:--------------------------:|:----------:|:-----------:|:-----------------------:|:-----------------------:|:--------:|:-----:|:-------------:|
| CoLA  |       Matthews corr        |   63.81    |    63.23    |         242.865         |         203.362         |    O2    |   3   |      1.8      |
| SST-2 |          Accuracy          |    93.0    |    93.92    |         242.869         |         208.024         |    O2    |   3   |      1.8      |
| MRPC  |             F1             |   90.17    |    90.29    |         244.989         |         207.063         |    O2    |   5   |      1.8      |
| STS-B |        Spearman cor        |   88.92    |    89.49    |         246.16          |         201.876         |    O2    |   3   |      1.8      | 
|  QQP  |             F1             |   87.03    |    87.97    |         242.804         |         207.756         |    O2    |   3   |      1.8      |
| MNLI  | Matched acc/MisMatched acc | 86.71/86.8 | 86.46/86.09 |         245.052         |         207.809         |    O2    |   3   |      1.8      |
| QNLI  |          Accuracy          |   91.82    |    91.9     |         242.437         |         208.137         |    O2    |   3   |      1.8      |
|  RTE  |          Accuracy          |   65.34    |    68.75    |         242.835         |         204.021         |    O2    |   5   |      1.8      |
| WNLI  |          Accuracy          |    50.7    |    54.69    |         239.393         |         205.816         |    O2    |   1   |      1.8      |

**表 3**  8卡训练结果展示表

| TASK  |           Metric           | 8p-精度(竞品A)  | 8p-精度(NPU)  | 8p-性能(竞品A)<br/>sample/s | 8p-性能(NPU)<br/>sample/s | AMP_Type | Epoch | Torch_Version |
|:-----:|:--------------------------:|:-----------:|:-----------:|:-----------------------:|:-----------------------:|:--------:|:-----:|:-------------:|
| CoLA  |       Matthews corr        |    58.3     |    62.16    |         1474.32         |        1340.928         |    O2    |   3   |      1.8      |
| SST-2 |          Accuracy          |    92.78    |    93.0     |        1606.601         |        1490.631         |    O2    |   3   |      1.8      |
| MRPC  |             F1             |    84.86    |    87.23    |         666.44          |        1120.797         |    O2    |   5   |      1.8      |
| STS-B |        Spearman cor        |    85.74    |    87.43    |         876.796         |         1237.05         |    O2    |   5   |      1.8      |
|  QQP  |             F1             |    88.02    |    87.67    |        1694.026         |        1511.692         |    O2    |   3   |      1.8      |
| MNLI  | Matched acc/MisMatched acc | 86.36/86.36 | 85.19/85.33 |        1726.692         |        1511.887         |    O2    |   3   |      1.8      |
| QNLI  |          Accuracy          |    91.95    |    91.10    |        1700.906         |        1498.031         |    O2    |   3   |      1.8      |
|  RTE  |          Accuracy          |    58.12    |    65.34    |        1042.305         |        1010.221         |    O2    |   5   |      1.8      |
| WNLI  |          Accuracy          |    43.66    |    55.32    |         498.107         |         338.084         |    O2    |   1   |      1.8      |



# 版本说明

## 变更

2023.02.11：首次发布。

## FAQ
   - 由于某些数据集较小，在进行8p训练时，竞品和NPU的精度均会较1p训练出现一定程度上的下降。
   - 因sklearn自身bug，若运行环境为ARM，则需要手动导入so，以下是root python环境里的示例

     ```export LD_PRELOAD=/usr/local/python3.7.5/lib/python3.7/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0```










