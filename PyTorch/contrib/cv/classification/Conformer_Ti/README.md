# Conformer_Ti for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

Conformer_Ti是一种新型的图像分类网络，由卷积神经网络（CNN）和注意力网络（Transformer）两个分类网络组成。另一个主要特征是FCU模块，该模块允许特征信息在两个学习网络之间交互。这些特征允许Conformer_Ti实现更好的分类性能。

- 参考实现：

  ```
  url=https://github.com/pengzhiliang/Conformer
  commit_id=815aaad3ef5dbdfcf1e11368891416c2d7478cb1
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
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装numactl：

  ```
  pt-get install numactl # for Ubuntu
  yum install numactl # for CentOS
  ```

- 安装依赖：

  ```
  pip install -r requirements.txt
  ```



## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集为ImageNet2012，将数据集上传到服务器任意路径下并解压。

   ImageNet2012数据集目录结构参考如下所示。

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
     bash ./test/train_full_1p.sh --data_path=real_data_path   
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=real_data_path  
     ```

   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data                                //数据集路径
   --addr                                //主机地址
   --model                               //使用模型，默认：Conformer_tiny_patch16
   --workers                             //加载数据进程数      
   --epoch                               //重复训练次数
   --batch-size                          //训练批次大小
   --lr                                  //初始学习率，默认：0.01
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。



# 训练结果展示

**表2**  训练结果展示表

| NAME    | Acc@1 |  FPS | Epochs | AMP_Type |
| :-----: | :---: | :--: | :----: | :------: |
| 1p-竞品 | -     |  241.5241 |   2   |        O1 |
| 1p-NPU1.5  | -     |  293.8400 | 2     |       O1 |
| 1p-NPU1.8  | -     |  237.3549 | 2     |       O1 |
| 8p-竞品 | 81.3 | 1712.3854 | 300    |        O1 |
| 8p-NPU1.5  | 81.4 | 2265.6700 | 300    |       O1 |
| 8p-NPU1.8  | 81.4 |  1820.9316 | 300     |       O1 |


# 版本说明

## 变更

2022.08.24：首次发布

2022.11.22：更新pytorch1.8版本，重新发布

## 已知问题

无。











