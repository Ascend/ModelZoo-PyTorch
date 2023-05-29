# FairMOT for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

FairMOT是一阶段多目标跟踪器（one-shot MOT），检测模型和Re-ID重识别模型同时进行，提升了运行速率。FairMOT采用 anchor-free 目标检测方法（CenterNet），估计高分辨率特征图上的目标中心和位置；同时添加并行分支来估计像素级 Re-ID 特征，用于预测目标的 id。

- 参考实现：

  ```
  url=https://github.com/ifzhang/FairMOT
  commit_id=815d6585344826e0346a01efd57de45498cfe52b
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

   请用户自行下载**MOT17**数据集，并在任意路径下新建dataset目录，将下载好数据集存放在该目录下并解压。
   数据集目录结构参考如下：

   ```
   MOT17
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train(empty)
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理（按需处理所需要的数据集）。

   首先生成标注文件，对/FairMOT/src/gen_labels_16.py文件进行修改，将该文件的**seq_root**参数修改为 dataset文件夹的目录+'/MOT17/images/train' ，例如：/root/dataset/MOT17/images/train。

   其次将文件中的label_root参数修改为 dataset文件夹的目录+'MOT16/labels_with_ids/train' ，例如/root/dataset/MOT17/labels_with_ids/train，然后在当前目录下执行以下命令：

   ```
   python3 gen_labels_16.py
   ```

   最后下载参考实现的模型源码，将下载的FairMOT/src下的data文件夹放至本模型的src目录下。

## 准备预训练权重

请用户自行下载[DLA-34 official]：CenterNet (Objects as Points) ，可根据参考实现的源码链接进行预训练模型的获取，上面提供了多种训练后的模型文件，并将下载好的预训练权重放到/FairMOT/models/目录下。


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
     bash ./test/train_eval_8p.sh --data_path=/data/xxx/
     ```
   
   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --load_model 						//加载预训练模型
   --data_cfg 							//指定数据配置文件   
   --world_size  						//加载数据进程数
   --batch_size  						//批次大小
   --lr       						    //初始学习率
   --use_npu    						//是否启用npu训练
   --use_amp   						//是否启用amp
   --num_epochs                    	//训练周期数
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|  NAME  | MOTA |   FPS   | Epochs | AMP_Type | Torch_Version |
| :----: | :--: | :-----: | :----: | :------: | :-----------: |
| 1p-竞品V |  -   | 10  |   30    |    -    |      1.5      |
| 8p-竞品V | 84.7 | 76 |   30   |    -    |      1.5      |
| 1p-NPU |  -   | 5.7818  |   1    |    O1    |      1.8      |
| 8p-NPU | 85.2 | 38.2117 |   50   |    O1    |      1.8      |


# 版本说明

2023.03.01：更新readme，重新发布。

2021.10.16：首次发布。

## FAQ

无。