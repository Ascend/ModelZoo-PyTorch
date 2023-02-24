# DeepMar for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

DeepMar是一个深度多属性联合学习模型，采用ResNet50做主干网络，把行人属性识别当做了多标签分类问题，并且能够很好地利用属性之间的关联关系。

- 参考实现：

  ```
  url=https://github.com/dangweili/pedestrian-attribute-recognition-pytorch.git
  commit_id=468ae58cf49d09931788f378e4b3d4cc2f171c22
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | pillow==8.4.0 |
  | PyTorch 1.8 | pillow==9.1.0 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。


## 准备数据集

1. 获取数据集。

   请用户自行获取训练数据集**peta**，上传到服务器任意路径下并解压，数据集目录结构参考如下所示。

   ```
   ├── peta
         ├──PETA.mat                 
         ├──images  
              ├── 图片1
              ├── 图片2
              ├── ...
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
     bash ./test/train_full_1p.sh --data_path=./dataset/peta/  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=./dataset/peta/  单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=./dataset/peta/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=./dataset/peta/  # 8卡性能
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --save_dir                          // 数据集路径
   --addr                              // 主机地址
   --npu                               // 单卡训练指定训练用卡号
   --workers                           // 加载数据进程数，默认：2 
   --total_epochs                      // 重复训练次数，默认150
   --batch_size                        // 训练批次大小
   --new_params_lr                     // 学习率，默认：0.001
   --finetuned_params_lr               // 最终学习率，默认：0.001
   --steps_per_log                     // 打印间隔， 默认是20
   --multiprocessing_distributed       // 是否使用多卡训练
   --device_list                       // 多卡训练指定训练用卡号，默认值：'0,1,2,3,4,5,6,7'
   --amp                               // 是否使用混合精度
   --loss_scale                        // 混合精度loss scale大小
   --opt_level                         // 混合精度类型
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 3**  训练结果展示表

|  NAME  | Acc@1 |   FPS   | Epochs | AMP_Type | Torch_Version |
| :----: | :---: | :-----: | :----: | :------: | :-----------: |
| 1p-NPU |   -   | 648.823 |   4    |    O2    |      1.8      |
| 8p-NPU | 76.52 | 4485.81 |  150   |    O2    |      1.8      |


# 版本说明

## 变更

2023.02.21：更新readme，重新发布。

2022.03.08：首次发布。

## FAQ


无。