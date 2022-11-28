# Transformer for PyTorch

- [概述](概述.md)

- [准备训练环境](准备训练环境.md)

- [开始训练](开始训练.md)

- [训练结果展示](训练结果展示.md)

- [版本说明](版本说明.md)

  

# 概述

## 简述

Transformer模型通过跟踪序列数据中的关系来学习上下文并因此学习含义。该模型使用全Attention的结构代替了LSTM，抛弃了之前传统的Encoder-Decoder模型必须结合CNN或者RNN的固有模式，在减少计算量和提高并行效率的同时还取得了更好的结果。

- 参考实现：

  ```
  url=https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer
  commit_id=be349d90738e543b4106a5492b8573fad2b72c24
  ```


- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/nlp
  ```

- 通过Git获取代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

- 通过单击“立即下载”，下载源码包。

  

# 准备训练环境

## 准备环境

- 当前模型支持的硬件、NPU固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套          | 版本                                                         |
  | ------------- | ------------------------------------------------------------ |
  | 硬件          | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | NPU固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN          | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch       | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |


- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```
  pip install -r requirements.txt
  ```

## 准备数据集

1. 获取数据集。

    本次训练采用的数据集来自于[WMT](https://www.statmt.org/wmt17/)(Workshop on Machine Translation)，用户需自行下载的数据集如下表所示，并需将数据集上传到源码包中的./examples/translation目录下并解压。

    **表 2**  数据集简介表

   | 来源  | 名称                                                         |
   | ----- | ------------------------------------------------------------ |
   | wmt13 | [training-parallel-europarl-v7.tgz](http://statmt.org/wmt13/training-parallel-europarl-v7.tgz) |
   | wmt13 | [training-parallel-commoncrawl.tgz](http://statmt.org/wmt13/training-parallel-commoncrawl.tgz) |
   | wmt17 | [training-parallel-nc-v12.tgz](http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz) |
   | wmt17 | [dev.tgz](http://data.statmt.org/wmt17/translation-task/dev.tgz) |
   | wmt14 | [test-full.tgz](http://statmt.org/wmt14/test-full.tgz)       |

   其中，前四项语料为训练集+验证集；最后一项语料为测试集。


2. 数据预处理。
   - 进入源码包根目录下执行下面脚本：`sh run_preprocessing.sh`。 
   
     
   

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
     ```

     启动单卡性能。
   
     ```
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/
     ```
   
   - 单机8卡训练
   
     启动8卡训练。
   
      ```
      bash ./test/train_full_8p.sh --data_path=/data/xxx/
      ```
     
     启动8卡性能。
     
     ```
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/
     ```
   

其中--data\_path参数填写`run_preprocessing.sh`中DATASET_DIR的路径。

模型训练脚本参数说明如下。
```
公共参数：
--data_path                         //数据集路径
--addr                              //主机地址
--arch                              //使用模型，默认：transformer_wmt_en_de    
--train_epochs                      //重复训练次数
--batch_size                        //训练批次大小
--lr                                //初始学习率，默认：0.0006
--weight_decay                      //权重衰减，默认：0.0
--amp                               //是否使用混合精度
多卡训练参数：
--device-id '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡   
```


# 训练结果展示

**表 3**  训练结果展示表

| 名字   | 精度    | 性能（fps） | torch版本 |
| ------ | ------- | ----------- | --------- |
| NPU-1p | -       | 614.5       | 1.5       |
| NPU-8P | 10.8895 | 4287.8      | 1.5       |
| NPU-1p | -       | 869.903     | 1.8       |
| NPU-8p | 10.8858 | 6326.66     | 1.8       |



# 版本说明

## 变更

2022.11.11：更新torch1.8版本，重新发布。

2021.01.12：首次发布。

## 已知问题

无。