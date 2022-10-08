# DenseNet201 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

DenseNet-201 是一个预训练模型，已经在 ImageNet 数据库的一个子集上进行了训练。 该模型接受了超过一百万张图像的训练，可以将图像分类为1000个对象类别（例如键盘，鼠标，铅笔和许多动物）。DenseNet模型，它的基本思路与ResNet一致，也是建立前面层与后面层的短路连接，不同的是，它建立的是前面所有层与后面层的密集连接。DenseNet还有一个特点是实现了特征重用。这些特点让DenseNet在参数和计算成本更少的情形下实现比ResNet更优的性能。

- 参考实现：

  ```
  url=https://github.com/pytorch/vision.git
  commit_id=585ce2c4fb80ae6ab236f79f06911e2f8bef180c
  code_path=torchvision/models/densenet.py
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



# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```
  pip install -r requirements.txt
  ```



## 准备数据集

1. 获取数据集。

   用户自行获取Imagenet数据集，将数据集上传到服务器任意路径下并解压。

   以Imagenet数据集为例，数据集目录结构参考如下所示。

   ```
   ├── ImageNet
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
     bash ./test/train_full_1p.sh --data_path=数据集路径    
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=数据集路径  
     ```

   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data-path                         //数据集路径
   --model                             //使用模型
   --workers                           //加载数据进程数      
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小
   --lr                           		//初始学习率，默认：0.8
   --momentum                          //动量，默认：0.9
   --weight-decay                      //权重衰减，默认：0.0001
   --amp                               //是否使用混合精度
   --loss_scale_value                  //混合精度lossscale大小
   --apex-opt-level                    //混合精度类型
   --distributed       				//是否使用多卡训练
   --devices_ids[]     				//多卡训练指定训练用卡
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。



# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1  | FPS      | Epochs | AMP_Type | Torch_version |
| ------- | ------ | :------- | ------ | :------- | ------------- |
| 1p-竞品 | -      | -        | -      | -        | -             |
| 8p-竞品 | -      | -        | -      | -        | -             |
| 1p-NPU  | -      | 455.016  | 1      | O2       | 1.5           |
| 1p-NPU  | -      | 451.908  | 1      | O2       | 1.8           |
| 8p-NPU  | 74.986 | 3537.046 | 40     | O2       | 1.5           |
| 8p-NPU  | 74.548 | 4079.179 | 40     | O2       | 1.8           |

备注：以1.5 torch version自验数据作为标杆数据。



# 版本说明

## 变更

2022.09.12：更新pytorch1.8版本，重新发布。

2021.10.09：首次发布。

## 已知问题

无。

