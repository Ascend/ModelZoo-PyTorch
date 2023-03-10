# YOLACT_plus for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

YOLACT++是用于实时实例分割的卷积神经网络，该网络将DCNv2和ResNet的残差单元结合，相比Yolact网络，一是在backbone中将简单的卷积网络替换成了可变性卷积层DCNv2，二是在计算精度时使用了一个maskiou网络，使mask的估计更加准确。

- 参考实现：

  ```
  url=https://github.com/dbolya/yolact.git
  commit_id=57b8f2d95e62e2e649b382f516ab41f949b57239
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

## 准备数据集

1. 获取数据集。

   用户可通过脚本获取训练数据集**coco2017**，在源码包根目录下执行以下命令。

   ```
   bash data/scripts/COCO.sh
   ```

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

1. 请用户根据需要自行获取预训练模型，将获取的预训练模型放至在源码包根目录下新建的`weights/`目录下，预训练模型对应的命名全称如下所示。

   ```
   Resnet101:  resnet101_reducedfc.pth
   Resnet50：  resnet50-19c8e357.pth
   Darknet53： darknet53.pth
   ```

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

     启动单卡评测。

     ```
     bash ./test/train_eval_1p.sh --data_path=/data/xxx/ --pth_path=ckpt_path
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   --pth_path参数填写训练生成的权重文件路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --batch_size                        //训练批次大小
   --data_path                         //数据集路径
   --momentum                          //动量
   --weight_decay                      //权重衰减
   --seed                              //随机数种子设置
   --learning_rate                     //初始学习率
   --max_iter                          //训练迭代数
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|  NAME  | Acc@1 |  FPS   | AMP_Type |
| :----: | :---: | :----: | :------: |
| NPU-1p |   -   | 3.153  |    O0    |
| NPU-8p | 33.49 | 14.677 |    O0    |

# 版本说明

## 变更

2023.03.10：更新内容，重新发布。

2020.07.08：首次发布。

## FAQ

无。