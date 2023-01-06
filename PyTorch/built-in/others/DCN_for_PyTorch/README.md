# DCN_for_PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

Deep & Cross Network（DCN），是谷歌和斯坦福大学在2017年提出的用于Ad Click Prediction的模型，通过结合DNN和cross network，更加高效的学习特定阶数的组合特征。相比于DNN，DCN的logloss更低，而且参数的数量将近少了一个数量级。

- 参考实现：

  ```
  url=https://github.com/shenweichen/DeepCTR-Torch
  commit_id=8265c75237e473c7f238fd6ba44cb09f55d1d9a9
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/others
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
  | 硬件       | [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
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

1. 下载数据集。
```
wget https://criteostorage.blob.core.windows.net/criteo-research-datasets/kaggle-display-advertising-challenge-dataset.tar.gz
```
2. 解压数据集。
```
tar -xvf kaggle-display-advertising-challenge-dataset.tar.gz
```

3. 解压后文件如下所示。
```
├── train.txt  # train数据集，含有标注
├── test.txt   # test数据集，没有标注，由于没有标注，这个数据集不使用
├── readme.txt # 说明
```

   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。

4. 数据集预处理。

将模型根目录下的criteo_preprocess.py拷贝到数据集目录，然后进行预处理。
```
python3 criteo_preprocess.py train.txt
```
运行上述脚本后，将在train.txt的同级目录下生成train_after_preprocess_trainval_0.93.txt和train_after_preprocess_test_0.07.txt两个文件。

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
   --addr                              //主机地址
   --npu_id                            //npu训练卡id
   --dist                              //8卡分布式训练
   --device_num                        //npu训练卡数
   --trainval_path                     //训练和验证数据集路径
   --test_path                         //测试数据集路径
   --lr                                //训练学习率
   --use_fp16                          //使用半精度进行训练
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| Type | AUC | FPS       | Epochs   | torch_version |
| :------: | :------:  | :------: | :------: | :------: |
| NPU-1p | 80.79 | 4589 | 2      | 1.5   |
| NPU-8p | 80.76 | 7009 | 2   | 1.5 |
| NPU-1p | - | 4773.2264 | 2 | 1.8 |
| NPU-8p | 80.75 | 11503.6671 | 2 | 1.8 |

# 版本说明

## 变更

2023.01.03：更新readme，重新发布。

2021.07.08：首次发布。

## 已知问题

无。