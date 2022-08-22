# VGG16 for PyTorch

-   [交付件基本信息](交付件基本信息.md)
-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 交付件基本信息
应用领域（Application Domain）：Image Classification

模型版本（Model Version）：1.1

修改时间（Modified）：2020.10.14

大小（Size）：1054.72MB

框架（Framework）：PyTorch_1.5.0

模型格式（Model Format）：pth

精度（Precision）：Mixed

处理器（Processor）：Ascend910

应用级别（Categories）：Research

描述（Description）：基于Pytorch框架的VGG-16图像分类网络训练


# 概述

## 简述

VGG由5层卷积层、3层全连接层、softmax输出层构成，层与层之间使用max-pooling分开，所有隐层的激活单元都采用ReLU函数。VGG模型是2014年ILSVRC竞赛的第二名，第一名是GoogLeNet。但是VGG模型在多个迁移学习任务中的表现要优于GoogLeNet。而且，从图像中提取CNN特征，VGG模型是首选算法。它的缺点在于，参数量有140M之多，需要更大的存储空间。

- 参考实现：

  ```
  url=https://github.com/pytorch/vision.git
  commit_id=605a67ad07e5d228bfc365fcb1317a553a3330e4
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone https://gitee.com/leng-xiongjin/ModelZoo-PyTorch.git       # 克隆仓库的代码
  cd PyTorch/contrib/cv/classification/Vgg16_ID1630_for_PyTorch       # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

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

2. 数据预处理（按需处理所需要的数据集）。

## 获取预训练模型（可选）

请参考原始仓库上的README.md进行预训练模型获取。将获取的bert\_base\_uncased预训练模型放至在源码包根目录下新建的“temp/“目录下。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /VGG16
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

   --data_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data                              //数据集路径
   --start-epoch                       //开始的轮数
   --batch-size                        //批大小
   --print-freq                        //打印频率
   --evaluate                          //是否在验证集上的评估模型
   --a                                 //使用模型，默认：resnet18
   --addr                              //主机地址
   --seed                              //使用随机数种子，默认：49
   --workers                           //加载数据进程数，默认：128
   --learning-rate                     //学习率
   --mom                               //动量，默认：0.9
   --resume                            //checkpoint的路径
   --weight-decay                      //权重衰减，默认：0.0001
   --seed                              //使用随机数种子，默认：49
   --workers                           //加载数据进程数，默认：4
   --learning-rate                     //学习率
   --epoch                             //重复训练次数
   --dist-backend                      //分布式后端
   --world-size                        //分布式训练节点数
   --rank                              //进程编号，默认：-1
   --multiprocessing-distributed       //是否使用多进程在多GPU节点上进行分布式训练
   --pretrained                        //是否使用预训练模型，默认True
   --dist-url                          //用于设置分布式训练的url
   --amp                               //是否使用混合精度
   --loss-scale                        //混合精度lossscale大小
   --opt-level                         //混合精度类型
   --device                            //使用设备为GPU或者是NPU
   --prof                              //是否使用profiling来评估模型的性能
   --stop-step-num                     //在指定stop-step数后终止训练任务
   
          
   多卡训练参数：
   --multiprocessing-distributed       //是否使用多卡训练
   --device-list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表
| NAME  | Acc@1  | FPS  | Epochs  | AMP_Type  | Torch  |
|---|---|---|---|---|---|
| 1p-竞品  | -  |  - | -  | -  | -  |
| 1p-竞品  | -  | -  | -  | -  | -  |
| 1p-NPU  | 60.349  | 3527.840  | 120  | O2  | 1.5  |
| 1p-NPU  | 61.423  | 3450.014  | 120  | O2  | 1.8  |
| 8p-竞品  | -  | -  | -  | -  | -  |
| 8p-竞品  | -  | -  | -  | -  | -  |
| 8p-NPU   | 70.169  | 13898.131  | 120  | O2  | 1.5  |
| 8p-NPU   | 70.134  | 13187.129  | 120  | O2  | 1.8  |

备注：一定要有竞品和NPU。

# 版本说明

## 变更

2022.08.22：更新内容，重新发布。

2022.03.18：首次发布。

## 已知问题

**_当前发行版本中存在的问题描述。_**