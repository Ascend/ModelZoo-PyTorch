# YOLOV4_ID0396 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

YOLO是一个经典的物体检测网络，将物体检测作为回归问题求解。YOLO训练和推理均是在一个单独网络中进行。基于一个单独的end-to-end网络，输入图像经过一次inference，便能得到图像中所有物体的位置和其所属类别及相应的置信概率。

- 参考实现：

  ```
  url=https://github.com/WongKinYiu/PyTorch_YOLOv4/tree/master
  commit_id=71cd8d7ddfa880e11fa12f6dcf84dcecab21c79b
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/detection
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.6.0；pillow==8.4.0 |
  | PyTorch 1.8 | torchvision==0.9.1；pillow==9.1.0 |
  | PyTorch 1.11 | torchvision==0.9.1；pillow==9.1.0；numpy==1.21.6 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  
  pip install -r 1.11_requirements.txt  # PyTorch1.11版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。


## 准备数据集

1. 获取数据集。

   用户自行获取coco数据集，包含images图片和annotations文件。其中images图片和annotations文件从**coco**官网获取，另外还需要labels图片。将获取后的数据集解压放置服务器的任意目录下(建议放到源码包根目录XXX/coco/下)。

   数据集目录结构参考如下所示：

   ```
   coco
     |-- annotations
     |-- images
        |-- train2017
        |-- val2017   
     |-- labels
        |-- train2017
        |-- val2017
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
     bash ./test/train_full_1p.sh --data_path=real_data_path  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=real_data_path  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=real_data_path  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=real_data_path  # 8卡性能
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                         //数据集路径
   --device_id                         //训练指定训练用卡
   --img-size                          //图像大小，默认：608
   --data                              //训练所需的yaml文件，默认：coco.yaml                  
   --cfg                               //训练过程中涉及的参数配置文件
   --weights                           //权重，默认：yolov4.pt
   --batch-size                        //训练批次大小，默认：32
   --epochs                            //重复训练次数，默认：300
   --amp                               //是否使用混合精度
   --loss_scale                        //混合精度lossscale大小，默认：128
   --opt-level                         //混合精度类型，默认：O1
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

| 名称   | Acc@1 | FPS   | Epochs | AMP_Type | Torch_Version |
| :----: | :--: | :-----: | :-----: | :-----: | :-----: |
| NPU-1p | - | 91.24 | 1 | O1 | 1.8 |
| NPU-8p | 0.398 | 713.68 | 300 | O1 | 1.8 |


# 版本说明

## 变更

2023.02.15：更新readme，重新发布。

2021.08.20：首次发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
