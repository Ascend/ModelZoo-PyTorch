# ResNet50_for_PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

ResNet是ImageNet竞赛中分类问题效果较好的网络，它引入了残差学习的概念，通过增加直连通道来保护信息的完整性，解决信息丢失、梯度消失、梯度爆炸等问题，让很深的网络也得以训练。ResNet有不同的网络层数，常用的有18-layer、34-layer、50-layer、101-layer、152-layer。ResNet50的含义是指网络中有50-layer，由于兼顾了速度与精度，目前较为常用。

- 参考实现：

  ```
  url=https://github.com/pytorch/examples.git
  commit_id=49e1a8847c8c4d8d3c576479cb2fe2fd2ac583de
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
  | 硬件       | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```
  pip install -r requirements.txt
  ```

## 准备数据集


1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，将数据集上传到服务器任意路径下并解压。

   以ImageNet数据集为例，数据集目录结构参考如下所示。

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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/
     ```

   - 多机多卡性能数据获取流程。

     ```
     bash ./test/train_performance_multinodes.sh --data_path=数据集路径 --batch_size=单卡batch_size*所有卡数 --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
     如若遇到逻辑卡号与物理卡号不一致的情况，请手动在./test/train_performance_multinodes.sh中添加传参，例如--device_list='0,1,2,3'
     ```

   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --amp                               //是否使用混合精度
   --data                              //数据集路径
   --addr                              //主机地址
   --seed                              //训练的随机数种子   
   --workers                           //加载数据进程数
   --learning-rate                     //初始学习率
   --weight-decay                      //权重衰减
   --print-freq                        //打印周期
   --dist-backend                      //通信后端
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小
   --benchmark                         //设置benchmark状态
   --dist-url                          //设置分布式训练网址
   --multiprocessing-distributed       //是否使用多卡训练
   --world-size                        //分布式训练节点数量
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| Type | Acc@1 | FPS       | Epochs   |
| :------: | :------:  | :------: | :------: |
| NPU-1p | -  | 1517.157 | 1      |
| NPU-8p | 77.193 | 10752.654 | 90     |
| GPU-1p | - | 2065 | 1 |
| GPU-8p | - | 14268 | 90 |

# 版本说明

## 变更

2022.12.27：更新readme，重新发布。

2021.07.08：首次发布。

## 已知问题

本模型单卡和多卡使用不同的脚本，脚本配置有差异， 会影响到线性度， 目前正在重构中。