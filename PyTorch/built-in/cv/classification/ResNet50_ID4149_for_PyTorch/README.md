# Resnet50 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

ResNet是由微软研究院的Kaiming He等四名华人提出，是ImageNet竞赛中分类问题效果较好的网络，它引入了残差学习的概念，通过增加直连通道来保护信息的完整性，解决信息丢失、梯度消失、梯度爆炸等问题，让很深的网络也得以训练，可以极快的加速神经网络的训练。ResNet有不同的网络层数，常用的有18-layer、34-layer、50-layer、101-layer、152-layer。ResNet18的含义是指网络中有18-layer。本文档描述的ResNet50是基于Pytorch实现的版本。


- 参考实现：

  ```
  url=https://github.com/pytorch/examples.git
  commit_id=e6cba0aa46b2a33b01207e1451e0cd10ca96c04c
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
| 固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
| CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial |
| PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)|

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip3.7 install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，CIFAR-10等，将数据集上传到服务器任意路径下并解压。

   Resnet18迁移使用到的ImageNet2012数据集目录结构参考如下所示。

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
     bash ./test/train_full_1p.sh --data_path="/data/xxx/" 
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path="/data/xxx/"
     ```

   --data_path参数填写数据集路径。
   
   
   - 多机多卡训练

     启动多机多卡训练。

     ```
     bash test/train_cluster.sh --data_path==xxx --batch_size="xxx" --lr=="xxx" --train_epochs="xxx" --world_size="xxx" --node_rank="xxx" --master_addr="xxx"
     ```

   --data_path参数填写数据集路径    
   --batch_size网络训练的batch size, 集群bs的设置推荐: 总卡数 * 512 
   --train_epochs网络训练周期    
   --world_size集群训练节点数    
   --node_rank集群训练节点ID，每个节点不一样    
   --master_addr集群训练主节点ip
   --lr集群训练学习率, 4机32卡训练学习率推荐四点多   
   
    
   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data                              //数据集路径
   --a                                 //使用模型，默认：resnet50
   --j                                 //加载数据进程数，默认：4
   --epochs                            //重复训练次数，默认90
   --b                                 //批大小
   --lr                                //学习率，默认0.2
   --world-size                        //分布式训练节点数
   --rank                              //进程编号，默认：-1
   --seed                              //使用随机数种子
   --gpu                               //使用的NPU的id
   --multiprocessing-distributed       //是否使用多进程在多GPU节点上进行分布式训练
   --amp                               //是否使用混合精度
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表
| NAME  | Acc@1  | FPS  | Epochs  | AMP_Type  | Torch  |
|---|---|---|---|---|---|
| 1p-NPU  |   | 1680  | 90  | O2  | 1.8  |
| 8p-NPU   | 76.63  | 11910  | 90  | O2  | 1.8  |


# 版本说明

## 变更

2022.12.14：首次发布。

## 已知问题

无。
