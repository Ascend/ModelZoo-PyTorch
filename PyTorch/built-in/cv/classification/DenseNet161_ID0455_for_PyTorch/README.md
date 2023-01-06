# DenseNet161 for PyTorch

-   [概述](#1)
-   [准备训练环境](#2)
-   [开始训练](#3)
-   [训练结果展示](#4)
-   [版本说明](#5)



# 概述<a id="1"></a>

## 简述

DenseNet-161是一个经典的图像分类网络，对于一个L层的网络，DenseNet共包含L\*（L+1）/2个连接，相比ResNet，这是一种密集连接，他的名称也由此而来，另一大特色为通过特征在channel上的连接来实现特征重用（feature reuse），这些特点让DenseNet在参数和计算成本更少的情形下实现比ResNet更优的性能。

- 参考实现：

  ```
  url=https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
  commit_id=585ce2c4fb80ae6ab236f79f06911e2f8bef180c
  ```
  
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境<a id="2"></a>

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                                               |
  |----------------------------------------------------------------------------------| ------------------------------------------------------------ |
  | 硬件 | [1.0.15.3](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)    |
  | NPU固件与驱动 | [20.0.0.3](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)    |
  | CANN       | [5.1.RC1.1](https://www.hiascend.com/software/cann/commercial?version=5.1.RC1.1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)                           |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集imagenet2012，将数据集上传到服务器并解压。

   数据集目录结构参考如下所示。

   ```
   ├── ImageNet2012
         ├──train
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...   
              ├──...                     
         ├──val  
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...              
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。


# 开始训练<a id="3"></a>

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
     bash ./test/train_full_1p.sh --data_path=real_data_path  # 1p精度    
     bash ./test/train_performance_1p.sh --data_path=real_data_path  # 1p性能
     ```
   
   - 单机8卡训练
   
     启动8卡训练。
   
     ```
     bash ./test/train_full_8p.sh --data_path=real_data_path  # 8p精度
     bash ./test/train_performance_8p.sh --data_path=real_data_path  # 8p性能 
     ```

   其中real_data_path参数填写数据集路径。

   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --data                              //数据集路径
   --addr                              //主机地址
   --arch                              //使用模型，默认：densenet161
   --workers                           //加载数据进程数      
   --epoch                             //重复训练次数
   --batch-size                        //训练批次大小
   --lr                                //初始学习率，默认：0.1
   --momentum                          //动量，默认：0.9
   --weight_decay                      //权重衰减，默认：0.0001
   --amp                               //是否使用混合精度
   --loss-scale                        //混合精度lossscale大小
   --opt-level                         //混合精度类型
   多卡训练参数：
   --multiprocessing-distributed       //是否使用多卡训练
   --device-list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。
   

# 训练结果展示<a id="4"></a>

**表 2**  训练结果展示表

|  Acc@1  | FPS  	| Npu_nums | Epochs | AMP_Type | Torch |
| :-----: | :--: 	| :------: | :----: | :------: | :---: |
|    -    | 464.66 	|    1     |   1    |    -     |  1.5  |
| 75.75   | 3257.51 |    8     |   90   |    O2    |  1.5  |
|    -    | 548.16 	|    1     |   1    |    O2    |  1.8  |
| 75.99   | 3732.03 |    8     |   90   |    O2    |  1.8  |


# 版本说明<a id="5"></a>

## 变更

2022.12.30：Readme整改。

## 已知问题

无。

