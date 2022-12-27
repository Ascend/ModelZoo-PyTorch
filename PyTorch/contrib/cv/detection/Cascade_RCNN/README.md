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

- 通过Git获取代码方法如下：

  ```
  git clone {url}        # 克隆仓库的代码
  cd {code_path}         # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

- 通过单击“立即下载”，下载源码包。




# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 硬件 | [1.0.12](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [21.0.3.1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.0.3](https://www.hiascend.com/software/cann/commercial?version=5.0.3) |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。


- 安装依赖。

  ```shell
   # 安装依赖包
   pip install -r requirements.txt
  ```

- 配置环境。

  安装detectron2

   ```
    source Cascade_RCNN/test/env_npu.sh
    cd Cascade_RCNN
    python3.7 setup.py build develop
   ```

## 准备数据集 & 预训练模型

   * 下载coco数据集

        放在任意目录中。如已有下载可通过设置环境变量DETECTRON2_DATASETS=“coco 所在数据集路径”进行设置，如 export DETECTRON2_DATASETS=/opt/npu/，则 coco 数据集放在 /opt/npu/ 目录中
   * 下载预训练模型

        下载预训练模型 R-101.pkl,
        configs/COCO-Detection/cascade_rcnn_R_101_FPN_1x.yaml配置文件中MODEL.WEIGHTS 设置为R-101.pkl的绝对路径

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

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。


    * 启动1p训练
    ```
    # training 1p performance
    bash ./test/train_performance_1p.sh --data_path=/data/xxx
    ```
    * 启动8p训练
    ```
    # training 8p accuracy
    bash ./test/train_full_8p.sh --data_path=/data/xxx

    # training 8p performance
    bash ./test/train_performance_8p.sh --data_path=/data/xxx
    ```
    * 启动评估脚本
    ```
    #test 8p accuracy
    bash test/train_eval_8p.sh --data_path=/data/xxx --pth_path=./output/model_final.pth
    ```
    * 启动模型微调
    ```
    # finetuning 1p
    bash test/train_finetune_1p.sh --data_path=/data/xxx --pth_path=./output/model_final.pth
    ```

   --data_path：数据集路径。

   --pth_path：训练过程中生成的权重文件。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --config-file                        //使用配置文件路径
   --device-ids                         //设备id
   --num-gpu                            //使用卡数量
   AMP					                //是否使用混合精度
   OPT_LEVEL					        //混合精度类型
   LOSS_SCALE_VALUE                     //混合精lossscale大小
   SOLVER.IMS_PER_BATCH             	//训练批次大小
   SOLVER.MAX_ITER				        //训练迭代次数
   SOLVER.STEPS                        //达到相应迭代次数时lr缩小十倍
   DATALOADER.NUM_WORKERS              //加载数进程数
   SOLVER.BASE_LR                      //学习率
   ```

   日志和权重文件保存在生成output目录下

# 训练结果展示

**表 2**  训练结果展示表

| 名称    |  FPS   |  Ap |
| :------: | :------: | :------: |
| 1p-竞品 | 10  | ----- |
| 1p-NPU  | 5 | -----|
| 8p-竞品 | 80 | 42.72 |
| 8p-NPU  | 42 | 42.445 |


# 版本说明

## 变更

2021.10.17：首次发布。

## 已知问题

无。








