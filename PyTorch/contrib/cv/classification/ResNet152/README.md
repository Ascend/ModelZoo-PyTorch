# ResNet152 for PyTorch\_Owner

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述
ResNet是ImageNet竞赛中分类问题效果较好的网络，它引入了残差学习的概念，通过增加直连通道来保护信息的完整性，解决信息丢失、梯度消失、梯度爆炸等问题，让很深的网络也得以训练。ResNet有不同的网络层数，常用的有18-layer、34-layer、50-layer、101-layer、152-layer。ResNet152的含义是指网络中有152-layer。本文档描述的ResNet152是基于Pytorch实现的版本。

- 参考实现：

  ```
  url=https://github.com/tensorflow/models/tree/r2.1.0/official/r1/resnet
  ```

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
  Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

  ```
  git clone https://gitee.com/ascend/ModelZoo-PyTorch.git    
  cd  ModelZoo-PyTorch/PyTorch/contrib/cv/classification/ResNet152
  ```

- 通过单击“立即下载”，下载源码包。

# 准备训练环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动 | [1.0.15](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC1](https://www.hiascend.com/software/cann/commercial?version=5.1.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)或[1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖（根据模型需求，按需添加所需依赖）。

  ```
  pip install -r requirements.txt
  ```

## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，CIFAR-10等，将数据集上传到服务器任意路径下并解压。

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

2. 数据预处理（按需处理所需要的数据集）。

## 获取预训练模型（可选）

请参考原始仓库上的README.md进行预训练模型获取。将获取的bert\_base\_uncased预训练模型放至在源码包根目录下新建的“temp/“目录下。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd ModelZoo-PyTorch/PyTorch/contrib/cv/classification/ResNet152
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=xxx
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=xxx
     ```

   --data\_path参数填写数据集根目录

   模型训练脚本参数说明如下。

    --addr                              //主机地址
    --workers                           //加载数据进程数 
    --learning-rate                     //初始学习率
    --mom                               //动量，默认：0.9
    --weight-decay                      //权重衰减，默认：0.0001
    --multiprocessing-distributed       //是否使用多卡训练
    --batch-size                        //训练批次大小
    --amp                               //是否使用混合精度
    --epoch                             //重复训练次数
    --seed                              //使用随机数种子，默认：49
    --rank                              //进程编号，默认：0
    --loss-scale                        //混合精度lossscale大小
    --opt-level                         //混合精度类型
    --device                            //使用设备为GPU或者是NPU
    --print-freq                        //打印频率
    --data                              //数据集路径

# 训练结果展示

| NAME     | Acc@1    | FPS       | Epochs   | AMP_Type | Torch  |
| :------: | :------: | :------:  | :------: | :------: |:------:|
| NPU-1P   |  -       | 319.265   |  1       | 02       | 1.5    |
| NPU-8P   | 78.259   | 3651.900  | 140      | O2       | 1.5    |
| NPU-1P   | 79.102   | 587.359   | 137      | O2       | 1.8    |
| NPU-8P   | 78.212   | 3470.902  | 137      | O2       | 1.8    |

# 版本说明

## 变更

2022.8.29：更新内容，重新发布。

## 已知问题

**_当前发行版本中存在的问题描述。_**

无。