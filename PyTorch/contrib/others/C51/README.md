# C51 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

C51是一种值分布强化学习算法，C51算法的框架依然是DQN算法，采样过程依然使用epsilon-greedy策略取期望贪婪，并且采用单独的目标网络。与DQN算法不同的是，C51算法的卷积神经网络不再是行为值函数，而是支点处的概率，C51算法的损失函数不再是均方而是KL散度。

- 参考实现：

  ```
  url=https://github.com/ShangtongZhang/DeepRL
  commit_id=3e47451ef6de4e3458ca00db9b4b724f71b0ac01
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/others
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version |      三方库依赖版本      |
  | :-----------: | :----------------------: |
  |  PyTorch 1.5  | torchvision==0.2.2.post3 |
  |  PyTorch 1.8  |    torchvision==0.9.1    |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  ```
  > **说明：** 
  > 只需执行一条对应的PyTorch版本依赖安装命令。

- 安装baselines

  ```
  git clone https://github.com/openai/baselines.git
  cd baselines
  pip install -e '.[all]'
  ```
  
  >注意：为成功安装baselines，请确保tensorflow版本大于1.14
  
- 安装mpi4py

  ```
  conda install mpi4py
  ```


## 准备数据集

1. 获取数据集。

   该模型不需要单独准备训练数据集，配置好环境后即可开始训练。
   
   

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash test/train_full_1p.sh   # 单卡精度
     
     bash test/train_performance_1p.sh  # 单卡性能
     ```

   - 单机单卡评测

     启动单卡评测。

     ```
     bash test/train_eval_1p.sh --pth_path=data/CategoricalDQNAgent-train_full_1p-xx.model ---status_path=data/CategoricalDQNAgent-train_full_1p-xx.stats
     ```
   
   --pth_path参数填写训练权重生成路径，目录层级参考上述示例。
   
   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --use_device                        //设置训练设备类型
   --device_id                         //设置训练卡ID
   --max_steps                         //设置迭代数
   --pth_path                          //设置权重加载路径
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| Acc@1    | FPS       | Npu_nums | steps   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -       | 99.4 step/s    | 1   | 4000000    | O1    |

# 版本说明

## 变更

2023.03.22：更新readme，重新发布。

2020.07.08：首次发布。

## FAQ

无。



# 公网地址说明

代码涉及公网地址参考 public_address_statement.md