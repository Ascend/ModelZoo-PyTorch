# SSD-MobileNetV1 for PyTorch 

* [概述](概述.md)
* [准备训练环境](开始训练.md)
* [开始训练](开始训练.md)
* [版本说明](版本说明.md)

# 概述

## 简述

MobileNetV1是基于深度级可分离卷积构建的网络。 MobileNetV1将标准卷积拆分为了两个操作：深度卷积和逐点卷积 。
SSD是一种one-stage的目标检测框架。SSD_MobileNetV1使用MobileNetV1提取有效特征，之后SSD通过得到的特征图的信息进行检测。

- 参考实现：

  ```
  url=https://github.com/qfgaohao/pytorch-ssd
  commit_id=f61ab424d09bf3d4bb3925693579ac0a92541b0d
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

   用户自行获取原始数据集，可选用的开源数据集包括VOCdevkit等，将数据集上传到服务器任意路径下并解压。

   以VOCdevkit数据集为例，数据集目录结构参考如下所示。

   ```
   |——VOCdevkit
        |——VOC2007（VOC2012）
            |——Annotations
            |——ImageSets
            |——JPEGImages
            |——SegmentationClass
            |——SegmentationObject
        |——test
            |——VOC2007
                |——Annotations
                |——ImageSets
                |——JPEGImages
                |——SegmentationClass
                |——SegmentationObject
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

## 获取预训练模型

请用户在源码包根目录下新建"models/"文件夹，下载所需的预训练模型**mobilenet_v1_with_relu_69_5.pth**，并将预训练模型放置在"models/"文件夹下。


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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/ --validation_data_path=real_validation_path  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/ --validation_data_path=real_validation_path  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/ --validation_data_path=real_validation_path  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/ --validation_data_path=real_validation_path  # 8卡性能
     ```

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval.sh --data_path=/data/xx/ --pth_path=real_pre_train_model_path  
     ```
     > **说明：** 评测脚本的--data_path参数填写验证集路径，需写到验证集一级目录。

   --data_path参数填写数据集路径，需写到数据集的一级目录；
   
   --validation_data_path参数填写测试集路径，需写到数据集的一级目录；
   
   --pth_path参数填写训练过程中生成的权重文件路径（默认存储在"models/"文件夹下）。
   
   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --datasets                          //数据集路径
   --addr                              //主机地址     
   --num_epochs                        //重复训练次数
   --batch_size                        //训练批次大小
   --lr                                //初始学习率
   --momentum                          //动量，默认：0.9
   --weight_decay                      //权重衰减，默认：0.0005
   --amp                               //是否使用混合精度
   --opt_level                         //混合精度类型
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2** 训练结果展示表

| NAME	       | 	ACC@1   |  FPS  |   Epochs  |AMP_Type|Torch_Version|
|-------------|----------|-------|-----------|--------|--------|
| 	1p-NPU  | 	0.67807 |	  346	|    1	|       O1| 1.8 |
|  8p-NPU	 |0.6849	  | 2657	    | 240	     | O2       | 1.8   |

# 版本说明

## 变更

2023.03.01：更新readme，重新发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md