# Attention U-Net for PyTorch
-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

Attention U-Net 将注意力机制应用于UNet分割网络中，可以实现对有关区域的关注以及对无关区域的忽略。注意力机制可以很好地嵌入到CNN框架中，能够提高模型性能并且不增加计算量。

- 参考实现：

  ```
  url=https://github.com/LeeJunHyun/Image_Segmentation.git
  commit_id=db34de21767859e035aee143c59954fa0d94bbcd
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/semantic_segmentation
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
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)|


- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行下载 [ISIC 2018 dataset](https://challenge2018.isic-archive.com/task1/training/) 原始数据集。注意，仅仅需要下载2018年的 Training Data 和 Training Ground Truth。本任务用到的 Training Data 和 Training Ground Truth 类别的压缩包分别为 ISIC2018_Task1-2_Training_Input.zip 和 ISIC2018_Task1_Training_GroundTruth.zip。

   下载完成并解压，将数据集放至在源码包根目录下新建的 ISIC/dataset 目录下或者在 dataset.py 中修改路径参数为数据集文件所在路径，然后运行 dataset.py 将数据集划分为三部分，分别用于 training、validation和test，三部分的比例是70%、10% 和20%。数据集总共包含2594张图片，其中1815用于training，259 用于validation，剩下的520用于testing。

   ```
   python dataset.py
   ```
   
   ISIC 2018 数据集包含各种皮肤病照片以及病灶分割图。以将数据集放置到“/dataset”目录下为例，原始的图片的训练集、验证集和测试集图片分别位于“train”、“valid”和“test”文件夹路径下，已进行分割的图片的训练集、验证集和测试集图片分别位于“train_GT”、“valid_GT”和”test_GT”文件夹路径下。数据集目录结构参考如下所示。
   
   ```
    dataset
    ├── train
    │   └── 图片1、2、3...
    ├── train_GT
    │   └── 图片1、2、3...
    ├── test
    │   └── 图片1、2、3...
    ├── test_GT
    │   └── 图片1、2、3...
    ├── valid
    │   └── 图片1、2、3...
    └── valid_GT
        └── 图片1、2、3...
   ```
   
   > **说明：** 
   > 数据集路径以用户自行定义的路径为准

# 开始训练

## 训练模型
1. 进入解压后的源码包根目录。

    ```
    cd /${模型文件夹名称} 
    ```

2. 运行训练脚本。

    ```bash
    # training 1p full，单p上运行150个epoch，运行时间大约10h
    bash ./test/train_full_1p.sh --data_path=real_data_path --epochs=150

    # training 1p performance，单p上运行10个epoch，运行时间大约30min
    bash ./test/train_performance_1p.sh --data_path=real_data_path --epochs=10

    # training 8p full，8p上运行150个epoch，运行时间约为2.5h
    bash ./test/train_full_8p.sh --data_path=real_data_path --epochs=150

    # training 8p performance，8p上运行10个epoch，运行时间约为5min
    bash ./test/train_performance_8p.sh --data_path=real_data_path --epochs=10
    ```

    real_data_path为用户数据集实际存放路径。

日志路径:

- 日志生成路径

```
test/output/
```
- 日志备份路径

```
test/backup/1p_full/			# 1p精度
test/backup/1p_performance/		# 1p性能
test/backup/8p_full/			# 8p精度
test/backup/8p_performance/		# 8p性能
```
   模型训练脚本参数说明如下：


    公共参数：
    --data                              //数据集路径
    --addr                              //主机地址
    --arch                              //使用模型
    --workers                           //加载数据进程数，（1p:32/8p:128）  
    --epoch                             //重复训练次数，默认150
    --batch-size                        //训练批次大小，（1p:16/8p128）
    --lr                                //初始学习率，（1p:0.0002/8p:0.0016）
    --momentum                          //动量，（momentum1:0.5/momentum2:0.999）
    --weight_decay                      //权重衰减，默认：0.0001
    --amp                               //是否使用混合精度
    --loss-scale                        //混合精度lossscale大小，默认dynamic
    --opt-level                         //混合精度类型
    多卡训练参数：
    --multiprocessing-distributed       //是否使用多卡训练
    --device-list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡


# 训练结果展示

**表 2**  训练结果展示表

| NAME   | Acc@1 |    FPS | Epochs | AMP_Type |
| ------ | ----- | -----: | ------ | -------: |
| 1p-1.5 | -     |     65 | 10     |       O2 |
| 1p-1.8 | -     | 138.54 | 10     |       O2 |
| 8p-1.5 | 0.899 |    500 | 150    |       O2 |
| 8p-1.8 | 0.95  | 845.93 | 150    |       O2 |


# 版本说明

## 变更

2022.09.26：更新pytorch1.8版本，重新发布。

2020.08.17：首次发布。

## 已知问题

无。
