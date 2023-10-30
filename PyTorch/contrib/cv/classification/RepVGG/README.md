# RepVGG PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述

## 简述
RepVgg是一个分类网络，该网络是在VGG网络的基础上进行改进的，主要改进点包括：在VGG网络的Block块中加入了Identity和残差分支，相当于把ResNet网络中的精华应用到VGG网络中；模型推理阶段，通过Op融合策略将所有的网络层都转换为Conv3*3，便于模型的部署与加速。

- 参考实现：

  ```
  url=https://github.com/DingXiaoH/RepVGG
  commit_id=9f272318abfc47a2b702cd0e916fca8d25d683e7
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
  ```

# 准备训练环境

## 准备环境

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

  ```
  pip install -r requirements.txt
  ```

## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，将数据集上传到服务器任意路径下并解压。 数据集目录结构如下所示：
   ```
   ├── ImageNet2012
         ├──train
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...
              ├──...
         ├──val
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...
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
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度

     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```
   - 单机8卡评测

     启动8卡评测
      ```
      bash ./test/train_eval_8p.sh --data_path=/data/xxx/  # 8卡评测
      ```

   - 单机单卡微调

     启动单1卡微调
      ```
      bash ./test/train_finetune_1p.sh --data_path=/data/xxx/  # 单卡微调
      ```

    --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   -a                                  // 网络结构名称
   --data                              // 数据集路径
   --workers                           // 数据读取并行量
   --epochs                            // 模型计算轮数
   --lr                                // 学习率
   --wd                                // 权重衰减参数
   --amp                               // 是否用apex
   --device                            // 指定设备类型
   --num_gpus                          // 使用几张卡
   --rank_id                           // 当前第几进程
   --addr                              // 集合通信地址
   --port                              // 集合通信端口
   --custom-weight-decay               // 自定义权重衰减
   --dist-backend                      // 指定分布式后台
   --opt-level                         // apex使用级别
   --loss-scale-value                  // apex缩放比例
   --batch-size                        // 训练批次大小
   ```

   训练完成后，权重文件保存在output文件夹下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V |   -   |   -     |   3    |    O2     |      1.5      |
| 8p-竞品V |   -   |    -    |  120   |    O2     |      1.5      |
|  1p-NPU-ARM  |   -   | 1686.26 |   3    |    O2    |      1.8      |
|  8p-NPU-ARM  | 69.43 | 12488.12 |  120   |    O2    |      1.8      |
|  1p-NPU-非ARM  |   -   | 2323.391 |   3    |    O2    |      1.8      |
|  8p-NPU-非ARM  | - | 8478.503 |  120   |    O2    |      1.8      |

# 版本说明

## 变更

2022.10.24：更新torch1.8版本，重新发布。

2021.07.13：首次发布

## 已知问题

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md