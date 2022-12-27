# Lightweight OpenPose for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述
## 简述

Lightweight_OpenPose是对原OpenPose模型的改进版。在基本思想方面，Lightweight_OpenPose的方法并未有太大的变动。Lightweight_OpenPose的目标是在cpu上实现实时的单图多目标的姿态估计任务。其主要方法是使用小而精的mobilenet作为backbone；使用预训练模型初始化参数；轻量的refinement模块；多refinement模块的训练方法

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
  | 硬件 | [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/)|

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```

## 准备数据集

1. 获取数据集。

   下载开源数据集包括coco2017，将数据集上传到服务器任意路径下并解压(假设路径名为<coco_home>)下。
   
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
   * 将训练标准文件转化为内部格式,在主目录下生成文件`prepared_train_annotation.pkl`
    ```shell
    python3.7.5 scripts/prepare_train_labels.py --labels <coco_home>/annotations/person_keypoints_train2017.json
    ```
   * 从完整的5000样本数量的验证集中随机生成一个样本量250的子集。在主目录下生成文件`val_subset.json`。
    ```shell
    python3.7.5 scripts/make_val_subset.py --labels <coco_home>/annotations/person_keypoints_val2017.json
    ```
2. 获取预训练的mobilenetv1权重文件

   下载`mobilenet_sgd_68.848.pth.tar`后将该文件放置在源码包根目录下。
   
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
     bash test/train_full_1p.sh --data_path=<coco_home> --step=1
     #step two,结果位于主目录下文件夹“step_two_checkpoints”
     bash test/train_full_1p.sh --data_path=<coco_home> --step=2
     #step three,结果位于主目录下文件夹“step_three_checkpoints”
     bash test/train_full_1p.sh --data_path=<coco_home> --step=3
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     # train 8p performance,结果位于主目录下文件夹“perf_8p_checkpoints”
     bash test/train_performance_8p.sh --data_path=<coco_home>

     # 8卡精度
     # train 8p full,模型经过三步step训练，依次执行以下脚本
     # step one,结果位于主目录下文件夹“step_one_checkpoints”
     bash test/train_full_8p.sh --data_path=<coco_home> --step=1
     #step two,结果位于主目录下文件夹“step_two_checkpoints”
     bash test/train_full_8p.sh --data_path=<coco_home> --step=2
     #step three,结果位于主目录下文件夹“step_three_checkpoints”
     bash test/train_full_8p.sh --data_path=<coco_home> --step=3
     ```
    - 验证阶段
      ```
       # 验证各阶段的最佳模型的精度，依次执行以下脚本
       # eval step one,结果位于主目录下文件夹“eval_step1”
       bash test/eval.sh --data_path=<coco_home> --step=1 --device_id=0 --checkpoint_path=./step_one_checkpoints/model_best.pth
       # eval step two,结果位于主目录下文件夹“eval_step2”
       bash test/eval.sh --data_path=<coco_home> --step=2 --device_id=1 --checkpoint_path=./step_two_checkpoints/model_best.pth
       # eval step three,结果位于主目录下文件夹“eval_step3”
       bash test/eval.sh --data_path=<coco_home> --step=3 --device_id=2 --checkpoint_path=./step_three_checkpoints/model_best.pth
      ```

   **训练的脚本需要在前一步骤结束后再接着启动。因为依赖于前一步保存的模型。验证的脚本使用单卡验证，所以训练完成后，可以分别启动三个脚本在不同卡上运行。单次验证时间约为3小时**

   --data_path：数据集路径

   --step：模型三步训练中第几步
   
   --device_id：指定卡号
   
   --checkpoint_path：已训练模型权重路径
   
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

# 训练结果展示

**表 2**  训练结果展示表

step-3阶段结果

|Name | Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------:| :------: | :------:  | :------: | :------: | :------: |
|1PGPU | -        | 254.017      | 1        | 1      | O1       |
|1PNPU | -        | 216.209      | 1        | 1      | O1       |
|8PGPU | 0.413        | 1228.599      | 8        | 280      | O1       |
|8PNPU | 0.4289     | 1800.749     | 8        | 280      | O1      |


8p-NPU上各阶段训练后的模型精度

| Acc@1    | step       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| 0.3973        | 1      | 8        | 280      | O1       |
| 0.4132     | 2     | 8        | 280      | O1      |
| 0.4289     | 3     | 8        | 280      | O1      |
# 版本说明

## 变更

2022.7.24：首次发布。

## 已知问题

无。

