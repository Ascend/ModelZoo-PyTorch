# Lightweight OpenPose for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述
## 简述

Lightweight_OpenPose是对原OpenPose模型的改进版。在基本思想方面，Lightweight_OpenPose的方法并未有太大的变动。Lightweight_OpenPose的目标是在cpu上实现实时的单图多目标的姿态估计任务。其主要方法是使用小而精的mobilenet作为backbone；使用预训练模型初始化参数；轻量的refinement模块；多refinement模块的训练方法。

- 参考实现：

  ```
  url=https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch.git
  commit_id=1590929b601535def07ead5522f05e5096c1b6ac
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/pose_estimation
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.2.2.post3 |
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

## 准备数据集

1. 获取数据集。

   用户自行下载 `coco2017` 数据集，将数据集上传到服务器任意路径下并解压(假设路径名为 `coco_home` )下。
   
   数据集目录结构参考如下所示。

   ```
    data
    ├── coco_home
        |── train2017  
                  ├──图片1 
                  │...           
                  ├──图片2           
        |── val2017  
                  ├──图片1 
                  │... 
                  ├──图片2
        ├── test2017 
                  ├──图片1 
                  │...
                  ├──图片2
        ├── annotations
                    ├person_keypoints_train2017.json 
                    ├person_keypoints_val2017.json

   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理。

   * 将训练标准文件转化为内部格式，在主目录下生成文件`prepared_train_annotation.pkl`。
     ```shell
     python3.7.5 scripts/prepare_train_labels.py --labels <coco_home>/annotations/person_keypoints_train2017.json
     ```
   * 从完整的5000样本数量的验证集中随机生成一个样本量250的子集。在主目录下生成文件`val_subset.json`。
     ```shell
     python3.7.5 scripts/make_val_subset.py --labels <coco_home>/annotations/person_keypoints_val2017.json
     ```

## 获取预训练模型

请用户自行获取预训练模型，将获取的 `mobilenet_sgd_68.848.pth.tar` 预训练模型放置在源码包根目录下。
   
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
     # 单卡性能
     bash ./test/train_performance_1p.sh --data_path=<coco_home>

     # 单卡精度
     # train 1p full,模型经过三步step训练，依次执行以下脚本
     # step one,结果位于主目录下文件夹“step_one_checkpoints”
     bash ./test/train_full_1p.sh --data_path=<coco_home> --step=1
     # step two,结果位于主目录下文件夹“step_two_checkpoints”
     bash ./test/train_full_1p.sh --data_path=<coco_home> --step=2
     # step three,结果位于主目录下文件夹“step_three_checkpoints”
     bash ./test/train_full_1p.sh --data_path=<coco_home> --step=3
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     # 8卡性能
     # train 8p performance,结果位于主目录下文件夹“perf_8p_checkpoints”
     bash ./test/train_performance_8p.sh --data_path=<coco_home>

     # 8卡精度
     # train 8p full,模型经过三步step训练，依次执行以下脚本
     # step one,结果位于主目录下文件夹“step_one_checkpoints”
     bash ./test/train_full_8p.sh --data_path=<coco_home> --step=1
     # step two,结果位于主目录下文件夹“step_two_checkpoints”
     bash ./test/train_full_8p.sh --data_path=<coco_home> --step=2
     # step three,结果位于主目录下文件夹“step_three_checkpoints”
     bash ./test/train_full_8p.sh --data_path=<coco_home> --step=3
     ```

   - 验证阶段

      启动单卡验证。
      ```
       # 验证各阶段的最佳模型的精度，依次执行以下脚本
       # eval step one,结果位于主目录下文件夹“eval_step1”
       bash ./test/eval.sh --data_path=<coco_home> --step=1 --device_id=0 --checkpoint_path=./step_one_checkpoints/model_best.pth
       # eval step two,结果位于主目录下文件夹“eval_step2”
       bash ./test/eval.sh --data_path=<coco_home> --step=2 --device_id=1 --checkpoint_path=./step_two_checkpoints/model_best.pth
       # eval step three,结果位于主目录下文件夹“eval_step3”
       bash ./test/eval.sh --data_path=<coco_home> --step=3 --device_id=2 --checkpoint_path=./step_three_checkpoints/model_best.pth
      ```

   > **说明：**
   >训练的脚本需要在前一步骤结束后再接着启动。因为依赖于前一步保存的模型。验证的脚本使用单卡验证，所以训练完成后，可以分别启动三个脚本在不同卡上运行。单次验证时间约为3小时。

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   --step：模型三步训练中第几步。
   
   --device_id：指定卡号。
   
   --checkpoint_path：已训练模型权重路径。
   
   模型训练脚本参数说明如下。

   ```
   公共参数：
   --from-mobilenet                          // 模型名称
   --train-images-folder                     // 数据集路径
   --prepared-train-labels                   // 预先生成的格式化文件
   --val-labels                              // 验证数据集标签
   --val-images-folder                       // 验证数据集目录
   --checkpoint-path                         // 已训练模型权重路径
   --base-lr                                 // 初始学习率
   --batch_size                              // 训练批次大小
   --print-freq                              // 打印频率
   --experiment-name                         // 训练步骤第几步
   --epochs                                  // 重复训练次数
   --addr                                    // 集合通信IP
   --rank                                    // 节点排名
   --dist-url                                // 分布式通信url
   --amp                                     // 使用混合精度
   --opt-level                               // apex优化器级别
   --loss-scale                              // apex loss缩放比例值
   --device                                  // 使用设备
   --num-workers                             // 加载数据线程数
   --gpu                                     // 使用卡号
   --world-size                              // 总进程数
   --dist-backend                            // 使用后台
   ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

step-3阶段结果

|   NAME   | Acc@1 |   FPS    | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :------: | :----: | :------: | :-----------: |
| 1p-竞品V |   -   | 254.017  |   1    |    O1     |      1.5      |
| 8p-竞品V | 0.413 | 1536.977 |  280   |    O1     |      1.5      |
|  1p-NPU-ARM  |   -   | 403.674  |   1    |    O1    |      1.8      |
|  8p-NPU-ARM  |  0.4289  | 2538.278  |  280   |    O1    |      1.8      |
|  1p-NPU-非ARM  |   -   | 475.648  |   1    |    O1    |      1.8      |
|  8p-NPU-非ARM  |  -  | 1214.179  |  280   |    O1    |      1.8      |

8p-NPU上各阶段训练后的模型精度

| Acc@1    | step       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| 0.3973        | 1      | 8        | 280      | O1       |
| 0.4132     | 2     | 8        | 280      | O1      |
| 0.4289     | 3     | 8        | 280      | O1      |

> **说明：**
> 以上精度结果为执行train_full_8p.sh脚本所得的验证集的结果。
# 版本说明

## 变更

2022.7.24：首次发布。

## 已知问题

无。

