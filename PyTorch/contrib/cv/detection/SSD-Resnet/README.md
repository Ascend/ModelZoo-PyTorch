# SSD-Resnet for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

SSD模型是用于图像检测的模型，通过基于Resnet34残差卷积网络(基础网络)，并向网络添加辅助结构，产生具有多尺度特征图的预测。在多个尺度的特征图中使用不同的默认框形状，可以有效地离散地输出不同大小的框，面对不同的目标可以有效地检测到，并且还可以对目标进行识别。

- 参考实现：

  ```
  url=https://github.com/mlcommons/training_results_v0.7 
  commit_id=585ce2c4fb80ae6ab236f79f06911e2f8bef180
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/detection
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
  > 只需执行一条对应的PyTorch版本依赖安装命令


## 准备数据集

1. 获取数据集。

   请用户自行准备数据集**coco**，可通过脚本进行获取，在源码包根目录下执行一下命令。

   ```
   source download_dataset.sh
   ```

   数据集目录结构参考如下所示。

   ```
   |-coco
   |-- annotations
   |   |-- captions_train2017.json
   |   |-- captions_val2017.json
   |   |-- instances_train2017.json
   |   |-- instances_val2017.json
   |   |-- person_keypoints_train2017.json
   |   |-- person_keypoints_val2017.json
   |-- train2017
   |-- val2017
   |-- test2017
   |   |-- 000000000001.jpg
   |   |-- 000000000016.jpg
   ```

   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。

### 获取预训练模型

请用户自行下载预训练模型**resnet34-333f7ec4.pth**，存放在源码包根目录下。

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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh --data_path=/data/xxx/ --checkpoint_path=real_pre_train_model_path
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   --checkpoint_path参数填写训练生成的权重文件路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data                              //数据集路径
   --batch-size                        //训练批次大小
   --loss_scale                        //loss scale大小
   --epochs                            //训练周期数
   --seed                              //随机数种子设置
   --lr                                //初始学习率
   --device_id                         //训练卡id设置
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|  NAME  |  Acc@1  |  FPS   | Epochs | AMP_Type | Torch_Version |
| :----: | :-----: | :----: | :----: | :------: | :-----------: |
| 1p-NPU |    -    | 351.35 |   1    |    O2    |      1.8      |
| 8p-NPU | 0.23509 |  1700  |   90   |    O2    |      1.8      |

# 版本说明

## 变更

2023.03.08：更新readme，重新发布。

2021.07.08：首次发布。

## FAQ

无。