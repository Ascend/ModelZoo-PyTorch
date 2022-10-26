
# FaceNet for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

FaceNet是一个用于目标识别网络，基于深度神经网络的图像映射方法和基于triplets的loss函数训练神经网络。该模型没有使用传统的softmax的方式去进行分类学习，而是抽取其中某一层作为特征，学习一个从图像到欧式空间的编码方法，然后基于该编码方法进行目标识别。本文档描述的FaceNet是基于Pytorch实现的版本。

- 参考实现：

  ```
  url=https://github.com/timesler/facenet-pytorch
  commit_id=2763e34b556d071a072e568892d2bdfd389c37b0
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/build-in/cv/classification
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取原始数据集。

   用户自行获取VGGFace2原始数据集，并解压放置任意目录下。
2. 对原始数据集进行数据预处理。

    使用源码包根目录下的 mtcnn_process.py 脚本对原始数据集中的“test”和“train”分别进行数据预处理。
在 mtcnn_process.py 脚本的rootdir参数中分别填写原始数据集中的“test”和“train”所在路径，
save_path参数中填写处理过后的数据集路径 （用户自行建立存储路径）。使用如下命令进行数据预处理。

    ```
    python3 mtcnn_process.py   # rootdir参数填写“test”所在路径
    python3 mtcnn_process.py   # rootdir参数填写“train”所在路径
    ```
    处理后的数据集目录结构如下所示：
    ```    
    ├── 用户自行建立的存储路径
       ├── VGG-Face2
          ├──test
               ├──n000001
                     │──图片1
                     │──图片2
                     │   ...       
               ├──n000002
                     │──图片1
                     │──图片2
                     │   ...
               ├──...                   
          ├──train  
               ├──n000001
                     │──图片1
                     │──图片2
                     │   ...       
               ├──n000002
                     │──图片1
                     │──图片2
                     │   ...
               ├──...
          ├──test_list.txt
          ├──train_list.txt
    ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

## 获取预训练模型

用户自行下载 [vggface2.pt](https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt) 预训练模型，并放到/root/.cache/torch/checkpoints/文件夹下。

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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/    
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/   
     ```

   --data_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_dir                              //数据集路径      
   --epoch                             //重复训练次数
   --batch_size                        //训练批次大小
   --lr                                //初始学习率，默认：0.01
   --amp_cfg                               //是否使用混合精度
   --loss_scale_value                        //混合精度lossscale大小
   --opt_level                         //混合精度类型
   多卡训练参数：
   --multiprocessing_distributed       //是否使用多卡训练
   --device_list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 |  FPS | Epochs | AMP_Type |PyTorch_version|
| ------- | ----- | ---: | ------ | -------: |-------: |
| 1p-NPU| -     |  2196.894 | 1      |        - |1.5
| 1p-NPU  |  -    | 2568.41  | 1      |       O2 |1.8.1
| 8p-NPU | 0.973 | 16251.8 | 8    |        - |1.5
| 8p-NPU  | 0.974 | 17267.7 | 8    |       O2 |1.8.1


# 版本说明

## 变更

2022.09.20：更新内容，重新发布。

2021.07.23：首次发布

## 已知问题
无。