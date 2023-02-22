# DLRM for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

DLRM（Deep Learning Recommendation Model）是深度学习推荐模型的实现，用于个性化推荐。该模型的输入分为稀疏特征和密集特征，同时该模型使用embedding来处理稀疏特征，使用多层感知机（MLP）来处理密集特征；并将这两个结果融合后再输入到多层感知机（MLP）中，来得到最终的结果。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/dlrm
  commit_id=adb39923b2e670bf8b7bde694de2a84396e818fa
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/others/
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 Pytorch 版本和已知已知三方库依赖如下所示。

  **表 1**  版本支持表
   
  | Torch_Version |    三方库依赖版本  |
  |---------------|-----|
  | Pytorch_1.8   | scikit-learn==0.20.4  |

  


- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  

- 安装依赖：

  ```
  pip install -r requirements.txt
  ```

- 安装mlperf-logging：

  ```
  git clone https://github.com/mlperf/logging.git mlperf-logging
  pip install -e mlperf-logging
  ```

## 准备数据集

1.获取数据集

  用户自行获取原始数据集，数据集为kaggle所提供的Criteo数据集，将获得的数据集上传到服务器的任意路径下载，数据集目录结构参考如下：

    ```
    $data_path
     └── train.txt
     └── test.txt
    ```
  > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。
  
2.数据集预处理
 
参照源码包目录下的data_utils.py文件进行数据集的预处理。


# 开始训练

## 训练模型
1. 进入解压后的源码包根目录
    ```
     cd /${模型文件名称} 
     ```
2. 运行训练脚本。

   该模型支持单机单卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=$data_path  # 单卡精度训练
     bash ./test/train_performance_1p.sh --data_path=$data_path  # 单卡性能 
     ```

     `--data_path`参数填写数据集的路径;

    - 模型训练脚本参数说明如下。

      ```
      公共参数：
      --test-freq                     //每多少step进行eval
      --loss-function                 //损失函数
      --learning-rate                 //学习率 
      --mini-batch-size               //batchsize
      --print-freq                    //每多少step打印一次
      --nepochs                       //训练的epoch数
      --local_rank                    //使用哪张卡进行训练
      --use-npu                       //是否使用NPU进行训练
      ```
 
# 训练结果展示

**表 2**  训练结果展示表

| NAME   | ACC    | FPS      | Epoch | AMP_Type | Torch_Version |
|--------|--------|----------|-------|----------|---------------|
| 1p-竞品V | 0.8024 | 8344.198 | 1     | O1       | 1.5           |
| 1p-NPU | 0.8024 | 3655.054 | 1     | O1       | 1.8           |






# 版本说明

## 变更

2022.03.18：首次发布。

2023.02.21: 更新内容，重新发布。

## FAQ
   无。










