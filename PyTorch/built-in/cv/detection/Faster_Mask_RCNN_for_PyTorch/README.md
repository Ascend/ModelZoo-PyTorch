# Faster Mask RCNN for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

FasterRCNN是一个业界领先的目标检测网络，他继承了FastRCNN的候选区域+目标识别架构，并在其基础上提出了候选区域网络（RPN）这一概念。通过共享全图卷积特征，FasterRCNN成功做到了让RPN不带来额外时间开销；而RPN的引入则将时下流行的神经网络“注意力”机制引入到了目标检测网络中。这些特性让FasterRCNN在ILSVRC以及COCO 2015等一系列竞赛上收获了第一名的成绩，同时在VGG-16等模型上拥有5fps的高速率。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/detectron2.git
  commit_id=be792b959bca9af0aacfa04799537856c7a92802
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/detection
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.6.0 |
  | PyTorch 1.8 | torchvision==0.9.1 |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。

- 源码编译 `Detectron2` 。

  编译器版本：gcc & g++ ≥ 5
  ```
  python3 -m pip install -e Faster_Mask_RCNN_for_PyTorch
  ```
  > **说明：** 
  >在重装PyTorch之后，通常需要重新编译detectron2。重新编译之前，需要使用`rm -rf build/**/*.so` 删除旧版本的build文件夹及对应的.so文件。

## 准备数据集

1. 获取数据集。

   用户自行获取原始 `COCO` 数据集，将数据集上传到服务器任意路径下并解压。

   数据集目录结构参考如下所示。

   ```
    ├── coco2017
    │   ├── annotations
    │          ├── captions_train2017.json
    │          ├── captions_val2017.json
    │          ├── instances_train2017.json
    │          ├── instances_val2017.json
    │          ├── person_keypoints_train2017.json
    │          ├── person_keypoints_val2017.json
    │   ├── train2017
    │          ├── 000000000009.jpg
    │          ├── 000000000025.jpg
    │          ├── ......
    │   ├── val2017
    │          ├── 000000000139.jpg
    │          ├── 000000000285.jpg
    │          ├── ......             
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

## 获取预训练模型

模型脚本会自动下载预训练权重文件。若下载失败，请自行准备 `R-101.pkl` 权重文件，将权重文件放到数据集同级路径下。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。    
   mask_rcnn启动训练    
   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能  
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

   - 多机多卡性能数据获取流程
   
     ```shell
     1. 安装环境
     2. 开始训练，每个机器请按下面提示进行配置
     bash ./test/train_performance_multinodes.sh --data_path=数据集路径 --batch_size=单卡batch_size*所有卡数 --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
     ```

   faster_rcnn启动训练     
   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_faster_rcnn_full_1p.sh --data_path=/data/xxx/  # 单卡精度
     bash ./test/train_faster_rcnn_performance_1p.sh --data_path=/data/xxx/  # 单卡性能  
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_faster_rcnn_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     bash ./test/train_faster_rcnn_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

   - 多机多卡性能数据获取流程
   
     ```shell
     1. 安装环境
     2. 开始训练，每个机器请按下面提示进行配置
     bash ./test/train_faster_rcnn_performance_multinodes.sh --data_path=数据集路径 --batch_size=单卡batch_size*所有卡数 --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
    AMP                                           //开启混合精度
    OPT_LEVEL                                     //设置混合精度优化等级为O2
    LOSS_SCALE_VALUE                              //设置损失函数缩放倍率为64
    MODEL.DEVICE                                  //指定运行脚本的物理设备
    SOLVER.IMS_PER_BATCH                          //指定输入batch中的图片张数
    SOLVER.MAX_ITER                               //指定最大训练迭代数（超过时训练终止）
    MODEL.RPN.NMS_THRESH                          //指定NMS阈值
    MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO      //指定BOX POOLER采样率
    MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO     //指定MASK POOLER采样率
    DATALOADER.NUM_WORKERS                        //指定DATALOADER所用进程个数
    SOLVER.BASE_LR                                //指定学习率
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

mask_rcnn结果
| NAME    | Acc@1 |  FPS  | Iters | AMP_Type | Torch_Version |
| :-----: | :---: | :---: | :----: | :------: | :-----------: |
| 1p-竞品V|   -   | - | 400      |     -    | 1.5 |
| 8p-竞品V|   -   | - | 10250    |     -    | 1.5 |
| 1p-NPU  |   -   | 6.071   | 400   |       O2 | 1.8 |
| 8p-NPU  | 26.3  | 42.933  | 10250 |       O2 | 1.8 |

faster_rcnn结果
| NAME    | Acc@1 |  FPS  | Epochs | AMP_Type | Torch_Version |
| :-----: | :---: | :---: | :----: | :------: | :-----------: |
| 1p-竞品V|   -   | - | 3000   |     -    | 1.5 |
| 8p-竞品V|   -   | - | 11250  |     -    | 1.5 |
| 1p-NPU  |     -   | 11.711  | 3000  |       O2 | 1.8 |
| 8p-NPU  |   26.6  | 88.901  | 11250 |       O2 | 1.8 |

# 版本说明

## 变更

2022.8.29：更新内容，重新发布。


## FAQ

无。
