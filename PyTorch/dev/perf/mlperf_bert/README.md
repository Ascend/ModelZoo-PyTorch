# mlperf_bert
-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

BERT的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的Encoder，是一种用于自然语言处理（NLP）的预训练技术。Bert-base模型是一个12层，768维，12个自注意头（self attention head）,110M参数的神经网络结构，它的整体框架是由多层transformer的编码器堆叠而成的。

- 参考实现：

  ```
  url=https://github.com/sharathts/training
  commit_id=f294d135a6b1ac12a19ea68c1f0e42e8acc39401
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/dev/perf/mlperf_bert
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表
  
  | Torch_Version | 三方库依赖版本 |
  |:-------:| :----------------------: |
  |  PyTorch 1.8  |    无    |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  1. 执行如下命令安装以来
  
  ```
  pip3 install -r requirements.txt
  ```
  2. 安装mlperf-logging
  
  下载mlperf-logging源码并安装 
  ```text
  git clone https://github.com/mlperf/logging.git mlperf-logging
  cd mlperf-logging
  pip3 install -e ./
  ```



## 准备数据集和预训练权重
1. 参考https://github.com/mlcommons、training_results_v1.0/tree/master/NVIDIA/benchmarks/bert/implementations/pytorch准备数据集合预训练权重
2. 把处理好的数据集合和预训练权重放到如下目录`<your_path>/input_preprocessing/`

   目录结构参考如下所示。

   ```
   $ your_path
       ├── input_preprocessing
           ├── 2048_shards_uncompressed
           │   ├── part_0_of_2048.hdf5
           │   ├── ...    
           │   └── part_2047_of_2048.hdf5
           ├── eval_set_uncompressed   
           │   └── part_eval_10k.hdf5
           ├── bert_config.json
           └── model.ckpt-28252.pt
   ```


# 开始训练

## 训练模型

1. 进入mlperf_bert目录。

   ```
   cd ./mlperf_bert
   ```

2. 执行`config.sh`脚本
   ```shell
   source config.sh
   ```
3. 切换到test目录
   ```shell
   cd ./pytorch/test
   ```
4. 运行训练脚本
   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ``` 
     bash train_performance_1p.sh --data_and_model_path=<your_path>/input_preprocessing     # 单卡性能
     ```
     
   - 单机8卡训练
     
     启动8卡训练。
        ```
        bash train_full_8p.sh --data_and_model_path=<your_path>/input_preprocessing
        ```


   模型训练脚本参数说明如下:
   ```
   公共参数：
   --data_and_model_path  // 指定数据集和预训练权重路径
   --fp32                 // 开启FP32模式
   ```
   训练完成后，权重文件保存在当前路径下，训练日志保存在`test/output/0/train_0.log`包含训练的精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Accuracy | FPS  | steps    |  Torch_Version |
|---------|----------|------|----------|:--------------|
| 1p-竞品A  | -        | 13.5 | 100      |  1.8           |
| 8p-竞品A  | 72       | 89   | 14000000 |  1.8           |
| 1p-NPU  | -        | 25.8 | 100      |  1.8           |
| 8p-NPU  | 72       | 185  | 14000000 |  1.8           |

**注：NPU数据采集自910B1**

# 版本说明

## 变更

2023.05.25: 更新训练结果。

2023.05.05: 首次发布。

## FAQ

- Q: 8卡训练完毕后主进程不退出。
   > PyTorch 1.8 上该模型源码也存在同样的问题，不影响训练和最终结果输出。 请在训练完毕后手动kill主进程。
 
