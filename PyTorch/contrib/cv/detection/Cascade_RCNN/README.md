# Cascade_RCNN for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述
Cascade R-CNN算法是CVPR2018的文章，通过级联几个检测网络达到不断优化预测结果的目的，与普通级联不同，Cascade R-CNN的几个检测网络是基于不同IOU阈值确定的正负样本上训练得到的。
- 参考实现：

  ```
  url=https://github.com/facebookresearch/detectron2.git
  commit_id=be792b959bca9af0aacfa04799537856c7a92802
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/detection
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

- 安装 `detectron2` 。
  ```
  source Cascade_RCNN/test/env_npu.sh
  cd Cascade_RCNN
  python3 setup.py build develop
  ```

## 准备数据集

1. 获取数据集。

   用户自行下载 `coco` 数据集，将数据集上传到服务器任意路径下并解压。
   
   数据集目录结构参考如下所示。
   ```
    ├── COCO
    │   │   ├── annotations
    |   |   │   │   ├── instances_val2017.json
    |   |   │   │   ├── instances_train2017.json
    |   |   │   │   ├── captions_train2017.json
    |   |   │   │   ├── ……
    │   │   ├── images
    |   |   │   │   ├──train2017
    |   |   |   |   │   │   ├──xxxx.jpg
    |   |   │   │   ├──val2017
    |   |   |   |   │   │   ├──xxxx.jpg
    │   │   ├── labels
    |   |   │   │   ├──train2017
    |   |   |   |   │   │   ├──xxxx.txt
    |   |   │   │   ├──val2017
    |   |   |   |   │   │   ├──xxxx.txt
    |   |   ├──test-dev2017.txt
    |   |   ├──test-dev2017.shapes
    |   |   ├──train2017.txt
    |   |   ├──……
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

## 获取预训练模型

用户自行下载 `R-101.pkl` 预训练模型，将获取的预训练模型放至在源码包根目录下，并将 `configs/COCO-Detection/cascade_rcnn_R_101_FPN_1x.yaml` 配置文件中 `MODEL.WEIGHTS` 设置为 `R-101.pkl` 的绝对路径。


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
     bash ./test/train_performance_1p.sh --data_path=/data/xxx  # 单卡性能
     ```
   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```
   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path # 8卡评测
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。
   
   --pth_path参数填写训练权重生成路径，需写到权重文件的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --config-file                       //使用配置文件路径
   --device-ids                        //设备id
   --num-gpu                           //使用卡数量
   AMP                                 //是否使用混合精度
   OPT_LEVEL                           //混合精度类型
   LOSS_SCALE_VALUE                    //混合精lossscale大小
   SOLVER.IMS_PER_BATCH                //训练批次大小
   SOLVER.MAX_ITER                     //训练迭代次数
   SOLVER.STEPS                        //达到相应迭代次数时lr缩小十倍
   DATALOADER.NUM_WORKERS              //加载数进程数
   SOLVER.BASE_LR                      //学习率
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Ap | FPS  | Iters | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :--------: |
| 1p-竞品V | - | 10 | 1000 | - | 1.5 |
| 8p-竞品V | 42.72 | 80 | 45000 | - | 1.5 |
| 1p-NPU | - | 6.16 | 1000 | O2 | 1.8 |
| 8p-NPU | 42.445 | 47.79 | 45000 | O2 | 1.8 |

# 版本说明

## 变更

2021.10.17：首次发布。

## FAQ

无。


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md





