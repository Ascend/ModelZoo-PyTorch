# Resnet18 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

ResNet是由微软研究院的Kaiming He等四名华人提出，是ImageNet竞赛中分类问题效果较好的网络，它引入了残差学习的概念，通过增加直连通道来保护信息的完整性，解决信息丢失、梯度消失、梯度爆炸等问题，让很深的网络也得以训练，可以极快的加速神经网络的训练。ResNet有不同的网络层数，常用的有18-layer、34-layer、50-layer、101-layer、152-layer。ResNet18的含义是指网络中有18-layer。本文档描述的ResNet18是基于Pytorch实现的版本。


- 参考实现：

  ```
  url=https://github.com/pytorch/examples.git
  commit_id=49e1a8847c8c4d8d3c576479cb2fe2fd2ac583de
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
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
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)|

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
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

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data                              //数据集路径
   --a                                 //使用模型，默认：resnet18
   --workers                           //加载数据进程数，默认：4
   --epochs                            //重复训练次数，默认90
   --start-epoch                       //开始训练轮数
   --batch-size                        //批大小，默认256
   --learning-rate                     //学习率，默认0.1
   --momentum                          //动量值，默认：0.9
   --weight-decay                      //权重衰减，默认：0.0001
   --print-freq                        //打印频率，默认：10
   --resume                            //checkpoint的路径
   --evaluate                          //是否在在验证集上评估
   --pretrained                        //是否使用预训练模型，默认True
   --world-size                        //分布式训练节点数
   --rank                              //进程编号，默认：-1
   --dist-url                          //用于设置分布式训练的url
   --dist-backend                      //分布式后端
   --seed                              //使用随机数种子
   --gpu                               //使用的GPU的id
   --multiprocessing-distributed       //是否使用多进程在多GPU节点上进行分布式训练
   --device                            //使用设备为GPU或者是NPU，默认NPU
   --addr                              //主机地址
   --device_list                       //设备id列表
   --amp                               //是否使用混合精度
   --loss-scale                        //混合精度lossscale大小
   --opt-level                         //混合精度类型
   --prof                              //是否使用profiling来评估模型的性能
   --stop-step-num                     //在指定stop-step数后终止训练任务
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表
| NAME  | Acc@1  | FPS  | Epochs  | AMP_Type  | Torch  |
|---|---|---|---|---|---|
| 1p-NPU  | 60.349  | 3527.840  | 120  | O2  | 1.5  |
| 1p-NPU  | 61.423  | 3531.6622  | 120  | O2  | 1.8  |
| 8p-NPU   | 70.169  | 13898.131  | 120  | O2  | 1.5  |
| 8p-NPU   | 70.049  | 17405.975  | 120  | O2  | 1.8  |


# 版本说明

## 变更

2022.08.22：更新内容，重新发布。

2022.03.18：首次发布。

## 已知问题

无。
