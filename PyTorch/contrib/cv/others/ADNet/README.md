# ADNet for PyTorch\_Owner

-   [交付件基本信息](交付件基本信息.md)
-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 交付件基本信息

应用领域（Application Domain）：Image denoising

模型版本（Model Version）：1.1

修改时间（Modified）：2022.06.30

_大小（Size）：_78kB

框架（Framework）：PyTorch\_1.8.1

模型格式（Model Format）：pth

精度（Precision）：O2

处理器（Processor）：Ascend 910

应用级别（Categories）：Research

描述（Description）：基于PyTorch框架的ADNet图像去噪网络训练

# 概述

## 简述

ADNet是一个注意力引导的图像去噪网络，它利用稀疏机制、特征增强机制和Attention机制在小网络复杂度的情况下提取显著性特征进而移除复杂图像背景中噪声。ADNet主要利用四个模块：一个稀疏块（SB），一个特征增强块（FEB）, 一个注意力机制（AB）和一个重构块(RB)来进行图像去噪。

- 参考实现：
```
 url=https://github.com/hellloxiaotian/ADNet.git 
```

- 适配昇腾 AI 处理器的实现：

```
url=https://gitee.com/ascend/ModelZoo-PyTorch.git 
code_path= ModelZoo-PyTorch/PyTorch/contrib/cv/others/ADNet

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
  | 固件与驱动 | [1.0.15](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC1](https://www.hiascend.com/software/cann/commercial?version=5.1.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)|

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖（根据模型需求，按需添加所需依赖）。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

请用户自行准备好数据集，包含训练集和验证集两部分，pristine_images_gray作为训练集，BSD68作为标签验证集。

   ```
          ADNET
└── data
|   └── BSD68
|   └── pristine_images_gray   
|   └── demo_img 
|       └──result
└── test      
└── dataset.py
└── demo.py
└── models.py
└── preprocess.py
└── test.py
└── train.py
└── utils.py
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理（按需处理所需要的数据集）。

source环境变量

```
source ./test/env_npu.sh
```

执行数据预处理脚本，将训练集图片裁剪成50*50的图片用与训练，运行成功会生成train.h5和val.h5文件。

```
python3 preprocess.py --preprocess True --mode S
```

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。
### 单p训练

source 环境变量

```
source ./test/env_npu.sh
```

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh   
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh   
     ```

日志路径:
    
    ./train_full_${num_gpus}.log           # training detail log
    ./train_performance_${num_gpus}.log  # 8p training performance result log
    ./eval_1p.log   # 8p training accuracy result log

   模型训练脚本参数说明如下。

   ```
--preprocess                       //是否在训练中预处理数据集，默认：False 
--batchSize                        //训练的batchsize，默认：128 
--resume                            //是否断点训练，默认：False 
--num_of_layers                   //网络总层数，默认：17       
--epoch                             //重复训练次数 
--batch-size                       //训练批次大小 
--lr                                 //初始学习率，默认：0.001 
--logdir                            //保存log路径 
--milestone                         //权重衰减的epoch，默认：30 
--outf                               //权重的输出路径参数之一 
--loss-scale                        //混合精度lossscale大小 
--mode                                //去噪的类型，有监督S或无监督B，默认：S 
--noiseL                             //噪声等级，默认：15，实验用25
--val_noiseL                        //测试的噪声等级，默认：15，实验用25
--resume                             //是否断点训练，默认：False 
--is_distributed                   //是否分布式训练，默认：0       
--local_rank                        //默认：0
--num_gpus                           //训练所用显卡个数，1为单卡，8为8卡
--world_size                         //分布式训练的节点数，默认：-1，实验用8 
单卡训练参数： 
--is_distributed                     //是否使用多卡训练，0 
--DeviceID '0'                        //单卡训练指定训练用卡，默认：’0’
多卡训练参数： 
--is_distributed                     //是否使用多卡训练，1  
--DeviceID '0,1,2,3,4,5,6,7'      //多卡训练指定训练用卡

   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | PT版本|精度 |  FPS | Epochs | AMP_Type |
| ------- |---- |----- | ---: | ------ | -------: |
| 1p-竞品V | 1.5|-     |  -| 1      |        - |
| 1p-NPU  | 1.5|-     |  1305 | 1      |       O2 |
| 1p-NPU  | 1.8|-     |  1696| 1      |       O2 |
| 8p-竞品V | 1.5|29.25 | - | 70    |        - |
| 8p-NPU  | 1.5|28.49 | 7100 | 70    |       O2 |
| 8p-NPU  | 1.8|28.52 | 12012 | 70    |       O2 |


备注：竞品的FPS数据未提及，故FPS数据只展示了使用NPU下的pytorch1.5和1.8.1的FPS结果。

# 版本说明

## 变更

2022.08.15：更新内容，重新发布。

2022.06.30：首次发布。

## 已知问题

**_当前发行版本中存在的问题描述。_**

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
