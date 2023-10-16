# MedSAM for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [推理评估](#推理评估)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)



# 概述
这是对MedSAM官方仓的迁移，使其能在NPU上进行训练和推理。
## 简述

***Segment Anything Model (SAM)*** 是一个图像分割类模型，它在1100万张图像和11亿张掩码的数据集上进行了训练，在各种分割任务中具有强大零样本能力。它可用于为图像中的所有对象生成分割的MASK，也可额外通过给模型输入诸如点、框、文本等PROMPT生成更准确的分割MASK。


***Medical SAM (MedSAM)*** 项目是在SAM基础上的下游应用，该项目将病灶的医疗影像数据作为输入，训练一个用于分割不同器官病灶的模型。不同于SAM的是，使用该模型需要给出bbox以获得更准确的病灶分割MASK，且由于训练数据的形式是一个BBOX内只包含一个MASK，因此推理的时候每个框只会给出1个准确的病灶MASK。

- 参考实现：

  ```
  url=https://github.com/bowang-lab/MedSAM.git
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/semantic_segmentation/
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的PyTorch 如下表所示。

  **表 1**  版本配套表

  | Torch_Version |             三方库依赖版本              |
  | :-----------: | :-------------------------------------: |
  |  PyTorch 2.0.1  | - |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集

   用户自行下载原始数据集MICCAI FLARE2022，在源码包根目录下新建目录data,并将数据集解压至该目录，数据集目录结构参考如下所示：

   ```
   ├── FLARE22Train
         ├──images
              ├──FLARE22_Tr_0001_0000.nii.gz    
              ├──FLARE22_Tr_0002_0000.nii.gz
              ├──...                     
         ├──labels  
              ├──FLARE22_Tr_0001.nii.gz     
              ├──FLARE22_Tr_0002.nii.gz
              ├──...
   ```

2. 数据预处理

   安装cc3d：
   ```shell 
   pip install connected-components-3d
   ```

   修改pre_CT_MR.py脚本：
   ```python
   nii_path = "data/FLARE22Train/images"  # path to the nii images
   gt_path = "data/FLARE22Train/labels"  # path to the ground truth
   ```
   在源码包根目录下执行数据预处理脚本：

   ```shell
   python pre_CT_MR.py
   ```


## 获取预训练模型

下载原始SAM模型。用户可根据segment-anything中的[README.md](https://github.com/facebookresearch/segment-anything#model-checkpoints)下载“sam_vit_b”，“sam_vit_l”，“sam_vit_h”权重文件，并在源码包根目录下新建目录models，将SAM模型放入“models/”目录下。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录

   ```
   cd /${模型文件夹名称} 
   ```
   
2. 运行训练脚本

   支持多机多卡训练，以单机8卡为例：

     ```
     # 启动单机8卡训练
     bash ./train_multi_npus.sh  
     ```
   模型训练脚本参数说明如下。
   ```
   分布式训练参数：
    NNODES=1                                 // 节点个数
    NODE_RANK=0                              // 节点ID
    NPUS_PER_NODE=8                          // 每个节点上使用的NPU数量
    MASTER_ADDR=localhost                    // 主节点IP
    MASTER_PORT=6789                         // 主节点端口

   公共参数：
    --task_name                              // 任务名称
    --model_type                             // 模型规格（vit_b|vit_l|vit_h）
    --tr_npy_path                            // 数据集路径
    --checkpoint                             // 加载模型权重路径
    --work_dir                               // 训练工作目录，用于保存权重、日志等
    --num_epochs                             // 训练轮数
    --use_amp                                // 是否使用混合精度训练
    --batch_size                             // batch size
    --grad_acc_steps                         // 梯度累计步数
    --num_workers                            // 数据集并行处理数
    --nnodes                                 // 节点个数
    --node_rank                              // 节点ID
    --nproc_per_node                         // 每个节点上的NPU数量
    --init_method                            // 分布式初始化
    ```
   
   训练完成后，权重文件保存在work_dir下，并输出模型训练loss等信息。

# 推理评估

## 评估模型

1. 进入解压后的源码包根目录

   ```
   cd /${模型文件夹名称} 
   ```
2. 转换模型
    通过训练得到的模型，需要通过转换脚本转换成通用格式才能进行推理，需要修改utils/ckpt_convert.py里的三个路径：

    ```shell
    # 原始的SAM模型权重路径
    sam_ckpt_path = "../models/sam_vit_l_0b3195.pth"
    # 训练好的MedSAM模型权重路径
    medsam_ckpt_path = "workdir/MedSAM-ViT-L-20230912-1734/medsam_model_latest.pth"
    # 转换后输出的模型权重路径
    save_path = "../models/medsam_vit_l_train.pth"
    ```
    然后执行转换脚本得到转换后的模型：
    ```shell
    python utils/ckpt_convert.py
    ```
   
3. 运行评估脚本

   支持多机多卡推理，这里以单机8卡为例：

     ```
     # 启动单机8卡评估，注意每张卡都会评估切分好的独立的一份测试集
     bash ./eval_multi_npus.sh  
     ```
   模型训练脚本参数说明如下。
   ```
   分布式训练参数：
    NNODES=1                                 // 节点个数
    NODE_RANK=0                              // 节点ID
    NPUS_PER_NODE=8                          // 每个节点上使用的NPU数量
    MASTER_ADDR=localhost                    // 主节点IP
    MASTER_PORT=6789                         // 主节点端口

   公共参数：
    --task_name                              // 任务名称
    --model_type                             // 模型规格（vit_b|vit_l|vit_h）
    --data_path                              // 数据集路径
    --checkpoint                             // 加载模型权重路径
    --work_dir                               // 训练工作目录，用于保存权重、日志等
    --num_workers                            // 数据集并行处理数
    --nnodes                                 // 节点个数
    --node_rank                              // 节点ID
    --nproc_per_node                         // 每个节点上的NPU数量
    --init_method                            // 分布式初始化
   ```
    评估完成后，会打印DSC、MIOU等评估指标，并保存一张推理对比图到当前路径。

# 训练结果展示

**表 2**  训练结果展示表

| NAME      | DSC | MIOU | FPS | Epochs     | AMP_Type|
| :-------:   | :-----: | :-----: | :---: | :------: | :-------: |
| 8p-A100   | 95.9 | 92.4 | 0.76 | 100      |  O0 |
| 8p-昇腾910 | 96.2 | 92.8 | 0.81 | 100     | O0 |

# 版本说明

## 变更
2023.09.26：首次发布。

## 已知问题

无。











