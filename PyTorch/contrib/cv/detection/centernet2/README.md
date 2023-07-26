# centernet2 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

CenterNet2是一个较为新颖的目标检测网络，相比于传统的two stage目标检测方法，在第一阶段仅作背景和物体的区分，并且使用先进的单级检测网络作为CenterNet2第一阶段，由此产生的检测器比它们的一级和二级前体更快、更准确。

- 参考实现：

  ```
  url=https://github.com/xingyizhou/CenterNet2.git 
  commit_id=68c0a468254b013e1d08309cd7a506756120ca62
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

  | Torch_Version |          三方库依赖版本           |
  | :-----------: | :-------------------------------: |
  |  PyTorch 1.5  | torchvision==0.6.0；pillow==8.4.0 |
  |  PyTorch 1.8  | torchvision==0.9.1；pillow==9.1.0 |

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

- 安装detectron2

  在源码包根目录下执行以下命令，编译安装detectron2。

  ```
  python3 setup.py build develop
  ```

## 准备数据集

1. 获取数据集。

   请用户自行获取训练数据集**coco2017**，将下载好的数据集上传到服务器任意路径下并解压。
   数据集目录结构参考如下所示。

   ```
   ├── coco2017
   │   ├── annotations
   │          ├── captions_train2017.json
   │          ├── captions_val2017.json
   │          ├── instances_train2017.json
   │          ├── instances_val2017.json
   │          ├── person_keypoints_train2017.json
   │          ├── person_keypoints_val2017.json
   │   ├── train2017
   │          ├── 000000000009.jpg
   │          ├── 000000000025.jpg
   │          ├── ......
   │   ├── val2017
   │          ├── 000000000139.jpg
   │          ├── 000000000285.jpg
   │          │   ...         
   ```

   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。

## 获取预训练模型

1. 请用户自行下载预训练模型**R-50.pkl**，并修改源码包根目录下`projects/CenterNet2/configs/Base-CenterNet2.yaml`配置文件中MODEL.WEIGHTS参数，设置为存放预训练模型文件的绝对路径。


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

   - 单机单卡评测

     将源码包根目录下`projects/CenterNet2/configs/Base-CenterNet2.yaml`中的字段WEIGHTS修改为训练生成的权重`model_final.pth`的绝对路径，实例如下。

     ```
     WEIGHTS: "/home/CenterNet2/result/CenterNet2/CenterNet2_R50_1x/model_final.pth"
     ```

     启动单机单卡评测

     ```
     bash ./test/train_eval_1p.sh --data_path=/data/xxx/
     ```

   --data_path参数填写数据集路径，需写到数据集的上一级目录。(例：数据集路径为/home/coco，则--data_path=/home)

   模型训练脚本参数说明如下。

   ```
   公共参数：
   SOLVER.IMS_PER_BATCH                //训练批次大小
   SOLVER.MAX_ITER                     //迭代次数
   --config-file                       //配置文件路径
   --num-machines                      //设备数统计
   --eval-only                         //设置是否仅进行评测
   ```
   
   训练完成后，会在`result/CenterNet2/CenterNet2_R50_1x`目录下保存模型文件model_final.pth，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Accuracy | FPS  | AMP_Type |
| :------: | :------: | :--: | :------: |
| 1p-竞品V |    -     | 12.3 |    O1    |
| 8p-竞品V |  43.68   | 90.5 |    O1    |
|  NPU-1p  |    -     | 2.86 |    O1    |
|  NPU-8p  |   43.5   | 18.7 |    O1    |

# 版本说明

## 变更

2023.03.10：更新readme，重新发布。

2020.07.08：首次发布。

## FAQ

1. 在arm环境上无法通过pip直接安装0.6.0版本的torchvision，可通过源码编译安装，参考方法如下。

   ```
   git clone https://github.com/pytorch/vision
   cd vision
   git checkout v0.6.0
   python3 setup.py install
   ```

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md