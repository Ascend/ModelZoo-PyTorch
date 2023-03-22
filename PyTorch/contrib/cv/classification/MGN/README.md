# MGN for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

MGN（Multiple Granularity Network）是一个多分支的深度网络，采用了将全局信息和各粒度局部信息结合的端到端特征学习策略。

- 参考实现：

  ```
  url=https://github.com/GNAYUOHZ/ReID-MGN.git
  commit_id=f0251e9e6003ec6f2c3fbc8ce5741d21436c20cf
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.2.2.post3 |
  | PyTorch 1.8 | torchvision==0.9.1 |
  
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

   用户自行获取 `Market-1501` 数据集，将数据集上传到服务器任意路径下并解压。

   数据集目录结构参考如下所示。

   ```
   ├── market1501
         ├──bounding_box_test
         ├──bounding_box_train  
         ├──query              
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
     bash ./test/train_eval_8p.sh --data_path=/data/xxx/  # 启动评测脚本前，需对应修改评测脚本中的--weight参数，指定ckpt文件路径
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。
   
   --weight参数填写训练权重生成路径，需写到权重文件的一级目录。
   
   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --data_path                         //数据集路径
   --local_rank                        //训练设备卡号
   --npu                               //是否使用NPU进行训练
   --device_num                        //训练设备数量      
   --lr                                //初始学习率
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   |  mAP  | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V |   -   | 71.408 |   1    |   O2     |      1.5      |
| 8p-竞品V | 93.35 | 771.818 |  500   |   O2    |      1.5      |
|  1p-NPU  |   -   | 29.408 |   1    |    O2    |      1.5      |
|  8p-NPU  | 93.83 | 200.024 |  500   |    O2   |      1.5      |

# 版本说明

## 变更

2022.03.18：首次发布。

## FAQ

无。


