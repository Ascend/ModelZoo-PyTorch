# YOLOR for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)



# 概述

## 简述

YOLOR提出了一个统一的网络来同时编码显式知识和隐式知识，在网络中执行了kernel space alignment（核空间对齐）、prediction refinement（预测细化）和 multi-task learning（多任务学习），同时对多个任务形成统一的表示，基于此进行目标识别。

- 参考实现：

  ```
  url=https://github.com/WongKinYiu/yolor
  commit_id=b168a4dd0fe22068bb6f43724e22013705413afb
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

   用户可进入源码包根目录，执行以下命令，下载coco数据集。数据集信息包含图片、labels图片以及annotations。数据集下载完成后，默认存放在源码包根目下的data文件中，若用户自行下载数据集，请将下载好的数据集存放在该目录下并解压。

   ```
   cd /${模型文件夹名称}
   bash scripts/get_coco.sh
   ```
   
    coco数据集目录结构参考如下：

   ```
   coco
   |-- LICENSE
   |-- README.txt
   |-- annotations
   |   |-- instances_val2017.json
   |-- images
   |   |-- test2017
   |   |-- train2017
   |   |-- val2017
   |-- labels
   |   |-- train2017
   |   |-- train2017.cache3
   |   |-- val2017
   |   |-- val2017.cache3
   |-- test-dev2017.txt
   |-- train2017.cache
   |-- train2017.txt
   |-- val2017.cache
   |-- val2017.txt
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

   - 单机单卡评测

     启动单卡评测。

     ```
     bash ./test/evaluation_npu.sh
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型性能脚本参数说明如下。

   ```
   公共参数：
   --data                                  //数据集路径      
   --epochs                                //重复训练次数
   --batch-size                            //训练批次大小
   --device                                //训练设备类型
   --workers                               //数据加载线程数
   --weights                               //初始权重
   --world-size                            //分布式训练节点数
   --npu                                   //npu训练卡id设置
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型性能信息。

# 训练结果展示


  **表 2**  训练结果展示表

|   NAME   | Acc@1 |  FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :---: | :----: | :------: | :-----------: |
| 1p-竞品V |   -   |  13   |   1    |    -     |      1.5      |
| 8p-竞品V | 51.4  |  113  |  120   |    -     |      1.5      |
|  1p-NPU  |   -   | 16.5  |   5    |    O1    |      1.8      |
|  8p-NPU  | 51.6  | 138.7 |  300   |    O1    |      1.8      |


# 版本说明

## 变更

2023.02.28：更新readme，重新发布。

2021.07.23：首次发布

## FAQ

无。