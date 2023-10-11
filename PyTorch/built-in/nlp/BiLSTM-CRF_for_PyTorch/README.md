# BiLstm for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述
LSTM常常被用来解决序列标注问题。LSTM依靠神经网络超强的非线性拟合能力，在训练时将样本通过高维空间中的复杂非线性变换，学习到从样本到标注的函数，之后使用这个函数为指定的样本预测每个token的标注。

- 参考实现：

  ```
  url=https://github.com/luopeixiang/named_entity_recognition
  commit_id=acb18af835ecdaab353d8185a79d82031df4e828
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/nlp
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |
  | PyTorch 1.11   | - |
  | PyTorch 2.1   | - |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  ```
  pip install -r requirements.txt
  ```

## 准备数据集

1. 获取数据集。

   请用户自行获取数据集ResumeNER，并将获取好的数据集放在源码包根目录下新建的BiLstm/文件夹下，数据集目录结构如下所示：

   ```
   ├── BiLstm
      ├── ResumeNER
         ├── dev.char.bmes               
         ├── test.char.bmes
         ├── train.char.bmes
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

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
     bash ./test/train_full_1p.sh  # 单卡精度
     
     bash ./test/train_performance_1p.sh  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh  # 8卡精度
     
     bash ./test/train_performance_8p.sh  # 8卡性能
     ```


模型训练脚本参数说明如下。

```
公共参数：
--amp_opt_level                              //混合精度类型
--seed                                       //固定随机参数 
--distributed                                //是否使用多卡训练
--local_rank                                 //指定的训练用卡
--batch_size                                 //训练批次大小
--train_epochs                               //重复训练次数
```

# 训练结果展示

**表 2**  训练结果展示表

| NAME   | Acc@f1 |       FPS |  AMP_Type | Torch_Version |
|:------:|:------:|:---------:|:------:|:------:|
| 8p-竞品V | 0.9513 | 16,457.14 | O2 | 1.5 |
| 1p-竞品V | 0.9592 |  2,021.05 |        O2 |     1.5 |
|  1p-NPU  | 0.9502 |    209    |    O2    |   1.8 |
|  8p-NPU  | 0.9643 |   1572    |    O2    |    1.8 |

> **说明：** 
   >该模型的性能评测需带cache或者开启二进制。

# 版本说明

## 变更

2023.02.14：更新readme，重新发布。

2022.09.16：首次发布。

## FAQ

无。
