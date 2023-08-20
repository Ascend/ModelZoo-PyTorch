# PPO for Pytorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

近端策略优化算法（Proximal Policy Optimization， PPO）是一种新型的Policy Gradient算法。为解决Policy Gradient算法中步长难以确定的问题，PPO提出了新的目标函数可以在多个训练步骤实现小批量的更新，是目前强化学习领域适用性最广的算法之一。


- 参考实现：

  ```
  url=https://github.com/nikhilbarhate99/PPO-PyTorch
  commit_id=6d05b5e3da80fcb9d3f4b10f6f9bc84a111d81e3
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

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.11 | Box2D==2.3.2 Box2D-kengz==2.3.3 gym==0.15.4 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》搭建torch环境。
  
- 安装依赖。

  在模型根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```shell
  pip install -r requirements.txt  
  ```


## 准备数据集

无。


## 获取预训练模型

无。

# 开始训练

## 训练模型

本文以BipedalWalker-v2场景为例，展示训练方法，其余场景需要根据场景替换启动脚本中的超参等配置。

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     ```shell
     bash test/train_full_1p.sh  # 单卡训练
     ```
     
   - 单机单卡性能
   
     ```shell
     bash test/train_performance_1p.sh  # 单卡性能
     ```
   
   训练完成后，权重文件保存在`test/output`路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME        | FPS    | MAX Training TimeSteps | Average Reward |
| ----------- | ------ | ---------------------- | -------------- |
| 1p-竞品V    | 581.82 | 3000000                 | 198.66         |
| 1p-NPU-910B | 232.74 | 3000000                | 235.22         |


# 公网地址说明
无。

# 版本说明

## 变更

2023.08.20：首次发布。

## FAQ

无。
   
