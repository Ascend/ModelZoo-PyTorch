# SSD-MobilenetV2 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述
SSD网络是继YOLO之后的one-stage目标检测网络，是为了改善YOLO网络设置的anchor设计的太过于粗糙而提出的，其设计思想主要是应用多尺度多长宽比的密集锚点设计和特征金字塔。在本模型中用Mobilenet-V2代替其原始模型中的VGG网络。

MobileNet-V2网络是由google团队在2018年提出的，相比MobileNet-V1网络，准确率更高，模型更小。 该网络中的主要亮点 ：
Inverted Residuals （倒残差结构 ）；Linear Bottlenecks（结构的最后一层采用线性层）。

- 参考实现：

  ```
    url=https://github.com/qfgaohao/pytorch-ssd.git 
    commit_id=6d7f986c4ac07744bf8375d1d01f9493663c47df
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

   请用户自行获取原始数据集VOCtrainval_11-May-2012.tar、VOCtrainval_06-Nov-2007.tar、VOCtest_06-Nov-2007.tar，在服务器任意路径下建立一个文件夹，将3个数据集上传到该文件夹下并解压。
   
   新建的文件夹名称可以自定义，在这里命名为VOC，数据集目录结构参考如下所示：
   ```
    |-VOC
    |
    |———————— VOC2007_trainval
    |         |——————Annotations
    |         |——————ImageSets
    |         |——————JPEGImages
    |         |——————SegmentationClass
    |         |——————SegmentationObject
    |———————— VOC2012_trainval
    |         |——————Annotations
    |         |——————ImageSets
    |         |——————JPEGImages
    |         |——————SegmentationClass
    |         |——————SegmentationObject
    |———————— VOC2007_test
              |——————Annotations
              |——————ImageSets
              |——————JPEGImages
              |——————SegmentationClass
              |——————SegmentationObject
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。


## 获取预训练模型

请用户自行下载预训练模型**mb2-imagenet-71_8.pth**，并在源码包根目录下新建**models**文件夹，将下载好的预训练模型存放在该目录下。


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

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   --data_path                     //数据集路径
   --dataset_type                  //数据集种类
   --base_net                      //预训练模型存放路径
   --lr                            //学习率
   --num_epochs                    //训练周期数
   --validation_epochs             //验证周期数
   --checkpoint_folder             //模型权重保存路径
   --batch_size                    //批次大小
   --device                        //训练设备类型，npu或gpu
   --momentum                      //动量
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2** 训练结果展示表

|  NAME  | Acc@1 |  FPS   | Epochs | AMP_Type | Torch_Version |
| :----: | :---: | :----: | :----: | :------: | :-----------: |
| 1p-NPU |   -   | 65.69  |   10   |    O2    |      1.8      |
| 8p-NPU | 0.68  | 214.74 |  200   |    O2    |      1.8      |


# 版本说明

## 变更

2023.03.01：更新readme，重新发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md