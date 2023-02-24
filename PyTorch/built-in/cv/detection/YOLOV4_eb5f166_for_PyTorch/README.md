# YOLOV4_eb5f166 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

YOLO算法作为one-stage目标检测算法最典型的代表，其基于深度神经网络进行对象的识别和定位，运行速度很快，可以用于实时系统。YOLOV4是继YOLOV3系列之后，又一个基准模型。

- 参考实现：

  ```
  url=https://github.com/WongKinYiu/PyTorch_YOLOv4/tree/master
  commit_id=eb5f1663ed0743660b8aa749a43f35f505baa325
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

   用户自行获取coco数据集，包含images图片和annotations文件。其中images图片和annotations文件可从**coco**官网获取，另外还需自行获取**labels**图片。将获取后的数据集解压放置服务器的任意目录下(建议放到源码包根目录XXX/coco/下)。

   数据集目录结构参考如下所示。

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
   
   用户还需自行获取VOC数据集，包含VOCtrainval_06-Nov-2007.zip、VOCtest_06-Nov-2007.zip、VOCtrainval_11-May-2012.zip，可从**VOC**官网获取。将获取后的数据集解压放置服务器的任意目录下，得到名为VOCDevkit的目录。

   数据集目录结构如下所示：

   ```
   VOCDevkit
      |-- VOC2007
      |-- VOC2012
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

     ```shell
     bash ./test/train_full_1p.sh --data_path=real_data_path  # COCO数据集，单卡精度    
     bash ./test/train_performance_1p.sh --data_path=real_data_path  # COCO数据集，单卡性能
     bash ./test/train_full_voc_1p.sh --data_path=real_data_path  # VOC数据集，单卡精度    
     bash ./test/train_performance_voc_1p.sh --data_path=real_data_path  # VOC数据集，单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```shell
     bash ./test/train_full_8p.sh --data_path=real_data_path  # COCO数据集，8卡精度    
     bash ./test/train_performance_8p.sh --data_path=real_data_path  # COCO数据集，8卡性能
     bash ./test/train_full_voc_8p.sh --data_path=real_data_path  # VOC数据集，8卡精度    
     bash ./test/train_performance_voc_8p.sh --data_path=real_data_path  # VOC数据集，8卡性能
     ```
   

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                         //数据集路径
   --workers                           //dataloader读数据线程数
   --batch-size                        //训练批次大小，默认32
   --data                              //训练所需的yaml文件             
   --cfg                               //训练过程中涉及的参数配置文件
   --img                               //训练图像大小，默认640 640
   --epochs                            //重复训练次数，默认：300
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表，COCO数据集，单卡32 batch size

| NAME     | mAP |  FPS | AMP_Type |
| -------  | -----  | ---: | -------: |
| 1p-竞品A  | - | 79.04 |       O1 |
| 8p-竞品A  | 0.480 | 568.32 |       O1 |
| 1p-NPU   | - | 95.67 |       O1 |
| 8p-NPU   | 0.480 | 721.92 |       O1 |

**表 3**  训练结果展示表，VOC数据集，单卡16 batch size

| NAME     | mAP50 |  FPS | AMP_Type |
| -------  | -----  | ---: | -------: |
| 1p-竞品A  | - | 71.11 |       O1 |
| 8p-竞品A  | 0.854 | 491.32 |       O1 |
| 1p-NPU   | - | 83.12 |       O1 |
| 8p-NPU   | 0.859 | 603.5 |       O1 |


# 版本说明

## 变更

2023.02.22：更新readme，重新发布。

2022.11.30：首次发布。

## FAQ

无。
