# Attention R2U-Net for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

Attention R2U-Net是基于U-Net模型的循环残差卷积神经网络 (RRCNN)。所提出的模型利用了U-Net、Residual Network以及RCNN的强大功能。这些提议的架构对于分割任务有几个优点。首先，残差单元有助于训练深度架构。第二，具有循环残差卷积层的特征积累确保了分割任务更好的特征表示。第三，它允许我们设计更好的U-Net架构，具有相同数量的网络参数，具有更好的医学图像分割性能。所提出的模型在三个基准数据集上进行了测试，例如视网膜图像中的血管分割、皮肤癌分割和肺病变分割等。

- 参考实现：

  ```
  url=https://github.com/LeeJunHyun/Image_Segmentation
  commit_id=db34de21767859e035aee143c59954fa0d94bbcd
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/semantic_segmentation
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

   用户自行下载  **ISIC 2018 dataset**原始数据集。 注意，仅仅需要下载2018年的Training Data和Training Ground Truth。下载完成并解压，将数据集放至在源码包根目录下新建的ISIC/dataset目录下或者在dataset.py中修改路径参数为数据集文件所在路径，然后运行dataset.py将数据集划分为三部分，分别用于training，validation 和 test 三部分的比例是70%，10% 和 20%。数据集总共包含2594张图片， 其中1815用于training，259 用于validation，剩下的520用于testing。

   本任务用到的 Training Data 和 Training Ground Truth 类别的压缩包分别为 ISIC2018_Task1-2_Training_Input.zip 和 ISIC2018_Task1_Training_GroundTruth.zip。

   ```
   python dataset.py
   ```

   ISIC 2018 数据集包含各种皮肤病照片以及病灶分割图。以将数据集放置到/dataset目录下为例，原始图片的训练集、验证集和测试集图片分别位于“train/”和“valid/”和test/”文件夹路径下，已进行分割的图片的训练集、验证集和测试集图片分别位于“train_GT/”和“valid_GT/”和test_GT/”文件夹路径下。
   
   数据集目录结构参考如下所示。
   
   ```
   ├── dataset
         ├──train
              │──图片1
              │──图片2
              │   ...       
   	  ├──train_GT
              │──图片1
              │──图片2
              ├──...                     
         ├──valid  
              │──图片1
              │──图片2
              ├──...  
         ├──valid_GT
              │──图片1
              │──图片2
              ├──... 
         ├──test  
              │──图片1
              │──图片2
              ├──...  
         ├──test_GT
              │──图片1
              │──图片2
              ├──...
   ```
   
   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。


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
   公共参数：
   --num_epochs                        //训练周期数
   --data_path                         //数据集路径 
   --batch-size                        //训练批次大小
   --loss-scale                        //设置loss scale大小
   --device_id                         //设置训练卡ID
   --use_apex                          //是否使用混合精度进行训练
   --apex_level                        //混合精度类型
   --world_size                        //训练进程数
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

|  NAME  | Acc@1 |  FPS   | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :----: | :----: | :------: | :----: |
| 1p-竞品V |  -   |   38   |  100   |    O2    | 1.5 |
| 8p-竞品V | 0.908 |  304   |  100   |    O2    | 1.5 |
|  1p-NPU  | 0.95  | 56.26  |  100   |    O2    | 1.8 |
|  8p-NPU  | 0.902 | 392.72 |  100   |    O2    | 1.8 |

# 版本说明

## 变更

2023.03.13：更新readme，重新发布。

2021.07.25：首次发布

## FAQ

1. 若0.6.0版本的torchvision版本无法直接安装，请参考源码进行安装。
   ```
   # 源码参考链接
   https://github.com/pytorch/vision
   ```
# 公网地址说明

代码涉及公网地址参考 public_address_statement.md