# MAPPO for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

多智能体近端策略优化算法（Multi-Agent Proximal Policy Optimization， MAPPO）是一种新型的Policy Gradient算法。基于现有的近端策略优化算法（Proximal Policy Optimization， PPO），在不修改算法架构的基础上，通过调整超参数，在多智能体环境中达到与大多数off-policy算法相当的性能。


- 参考实现：

  ```
  url=https://github.com/marlbenchmark/on-policy
  commit_id=b21e0f743bd4516086825318452bb6927a33538d
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/rl/
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      |                           三方库依赖版本                            |
  | :--------: |:------------------------------------------------------------:|
  | PyTorch 1.11 | absl-py==1.4.0; gym==0.17.2; protobuf==3.20.0; wandb==0.10.5 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》搭建torch环境。
  
- 安装依赖。

  在模型根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```shell
  pip install -r requirements.txt  
  pip install -e .
  ```


## 准备数据集

无。


## 获取预训练模型

无。

# 开始训练

## 训练模型

本文以MPE Comm场景为例，展示训练方法，其余场景需要根据场景替换启动脚本。

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练。

   - 单机单卡训练

     ```shell
     bash test/train_full_1p.sh  # 单卡训练
     ```
     
   - 单机单卡性能
   
     ```shell
     bash test/train_performance_1p.sh  # 单卡性能
     ```
   
   训练完成后，权重文件保存在`onpolicy/scripts/results`路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME      | FPS  | MAX Training TimeSteps | Average Reward |
|-----------|------|------------------------|----------------|
| 1p-竞品V   | 1800 | 2000000                | -15.86         |
| 1p-NPU    | 1271 | 2000000                | -15.82         |


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 版本说明

## 变更

2023.09.05：首次发布。

## FAQ

无。
   
