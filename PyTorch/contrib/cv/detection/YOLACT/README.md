# YOLACT for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

## 概述

### 简述

YOLACT是2019年发表在ICCV上面的一个实时实例分割的模型，它主要是通过两个并行的子网络来实现实例分割的。(1)Prediction Head分支生成各个anchor的类别置信度、位置回归参数以及mask的掩码系数；(2)Protonet分支生成一组原型mask。然后将原型mask和mask的掩码系数相乘，从而得到图片中每一个目标物体的mask。论文中还提出了一个新的NMS算法叫Fast-NMS，和传统的NMS算法相比只有轻微的精度损失，但是却大大提升了分割的速度。

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

  | Torch_Version | 三方库依赖版本 |
  | :-----------: | :------------: |
  |  PyTorch 1.5  | pillow==8.4.0  |
  |  PyTorch 1.8  | pillow==9.1.0  |

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

   请用户自行准备**coco**数据集，将数据集放置在服务器的任意目录下并解压。

   数据集目录结构参考如下所示：

   ```
   ├── coco
   	├── val2017/
   	├── train2017/
   	├── annotations/
   		├── instances_train2017.json
   		├── instances_val2017.json
   		├── ......
   ```

   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。


### 获取预训练模型

请用户自行下载预训练模型**resnet101_reducedfc.pth**，并在源码包根目录下新建`weights`文件夹，将下载好的预训练模型存放至该路径下。


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
     bash ./test/train_full_1p.sh /data/xxx/coco  # 单卡精度
     
     bash ./test/train_performance_1p.sh /data/xxx/coco  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh /data/xxx/coco  # 8卡精度
     
     bash ./test/train_performance_8p.sh /data/xxx/coco  # 8卡性能
     ```

   `/data/xxx/coco`参数填写coco数据集的路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --batch_size                        //训练批次大小
   --data                              //数据集路径
   --momentum                          //动量
   --seed                              //随机数种子设置
   --weight_decay                      //权重衰减
   --save_folder                       //权重保存文件
   --node_device                       //设置训练计算设备
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2** 训练结果展示表

| box mAP | mask mAP | FPS  | Npu_nums | Epochs | Steps  | AMP_Type |
| :-----: | :------: | :--: | :------: | :----: | ------ | -------- |
|  31.98  |  29.62   | 25.4 |    8     |   54   | 100000 | O0       |

#  版本说明

## 变更

2023.03.10：更新readme，重新发布。

2021.07.14：首次发布。

## FAQ

无。