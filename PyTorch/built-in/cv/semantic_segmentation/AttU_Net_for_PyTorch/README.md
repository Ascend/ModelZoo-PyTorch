# Attention U-Net for PyTorch
-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

Attention U-Net 将注意力机制应用于UNet分割网络中，可以实现对有关区域的关注以及对无关区域的忽略。注意力机制可以很好地嵌入到CNN框架中，能够提高模型性能并且不增加计算量。

- 参考实现：

  ```
  url=https://github.com/LeeJunHyun/Image_Segmentation.git
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

   用户自行下载 **ISIC 2018 dataset**原始数据集。注意，仅仅需要下载2018年的 Training Data 和 Training Ground Truth。本任务用到的 Training Data 和 Training Ground Truth 类别的压缩包分别为 ISIC2018_Task1-2_Training_Input.zip 和 ISIC2018_Task1_Training_GroundTruth.zip。

   下载完成并解压，将数据集放至在源码包根目录下新建的 ISIC/dataset 目录下或者在 dataset.py 中修改路径参数为数据集文件所在路径，然后运行 dataset.py 将数据集划分为三部分，分别用于 training、validation和test，三部分的比例是70%、10% 和20%。数据集总共包含2594张图片，其中1815用于training，259 用于validation，剩下的520用于testing。

   ```
   python dataset.py
   ```
   
   ISIC 2018 数据集包含各种皮肤病照片以及病灶分割图。以将数据集放置到“/dataset”目录下为例，原始的图片的训练集、验证集和测试集图片分别位于“train”、“valid”和“test”文件夹路径下，已进行分割的图片的训练集、验证集和测试集图片分别位于“train_GT”、“valid_GT”和”test_GT”文件夹路径下。数据集目录结构参考如下所示。
   
   ```
    dataset
    ├── train
    │   └── 图片1、2、3...
    ├── train_GT
    │   └── 图片1、2、3...
    ├── test
    │   └── 图片1、2、3...
    ├── test_GT
    │   └── 图片1、2、3...
    ├── valid
    │   └── 图片1、2、3...
    └── valid_GT
        └── 图片1、2、3...
   ```
   
   > **说明：** 
   > 数据集路径以用户自行定义的路径为准


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
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/ --epochs=10 # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/ --epochs=10  # 8卡性能
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下：

   ```
    公共参数：
    --batch-size                        //训练批次大小
    --image_size                        //图片大小
    --num_epochs                        //训练周期数
    --data_path                         //数据集路径
    --apex                              //是否使用混合精度进行训练  
    ---apex_opt_level                   //混合精度类型
    --seed                              //随机数种子设置
    --npu_idx                           //设置训练卡id
    --loss_scale_value                  //设置loss scale值
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

|  NAME  | Acc@1 |  FPS   | Epochs | AMP_Type | Torch_Version |
| :----: | :---: | :----: | :----: | :------: | :-----------: |
| 1p-NPU |   -   | 138.54 |   10   |    O2    |      1.8      |
| 8p-NPU | 0.95  | 845.93 |  150   |    O2    |      1.8      |


# 版本说明

## 变更

2023.03.13：更新readme，重新发布。

2020.08.17：首次发布。

## FAQ

无。
