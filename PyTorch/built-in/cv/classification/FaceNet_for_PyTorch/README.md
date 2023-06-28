
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

请用户自行下载 **vggface2.pt**预训练模型，并放到/root/.cache/torch/checkpoints/文件夹下。

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
   --seed                              //随机数种子设置     
   --epochs                            //重复训练次数
   --batch_size                        //训练批次大小
   --lr                                //初始学习率，默认：0.01
   --amp_cfg                           //是否使用混合精度
   --loss_scale_value                  //混合精度loss scale大小
   --opt_level                         //混合精度类型
   多卡训练参数：
   --multiprocessing_distributed       //是否使用多卡训练
   --device_list                       //多卡训练指定训练用卡
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 |  FPS | Epochs | AMP_Type |Torch_Version|
| :-----: | :---: | :--: | :----: | :------: |:------: |
| 1p-NPU  |  -    | 1910.06 | 1      |       O2 |1.8|
| 8p-NPU  | 0.974 | 15220.9 | 8    |       O2 |1.8|


# 版本说明

## 变更

2023.02.21：更新内容，重新发布。

2021.07.23：首次发布

## FAQ
无。


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
