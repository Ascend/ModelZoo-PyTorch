# R2U-Net for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述
R2U-Net是基于U-Net模型的循环残差卷积神经网络 (RRCNN)。所提出的模型利用了U-Net、Residual Network以及RCNN的强大功能。这些提议的架构对于分割任务有几个优点。首先，残差单元有助于训练深度架构。第二，具有循环残差卷积层的特征积累确保了分割任务更好的特征表示。第三，更好的 U-Net 架构，具有相同数量的网络参数，具有更好的医学图像分割性能。所提出的模型在三个基准数据集上进行了测试，例如视网膜图像中的血管分割、皮肤癌分割和肺病变分割等。

- 参考实现：

    ```
    url=https://github.com/LeeJunHyun/Image_Segmentation
    commit_id=db34de21767859e035aee143c59954fa0d94bbcd
    ```

- 适配昇腾 AI 处理器的实现：

    ```
    url=https://gitee.com/ascend/ModelZoo-PyTorch.git
    code_path=PyTorch/built-in/cv/semantic_segmentation
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
    | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
    | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
    | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 环境准备指导。

    请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

    ```
    pip install -r requirements.txt
    ```
## 准备数据集

1. 获取数据集。

    用户自行下载 [ISIC 2018 Dataset](https://challenge2018.isic-archive.com/task1/training/) 原始数据集。 注意，仅仅需要下载2018年的Training Data和Training Ground Truth。本任务用到的 Training Data 和 Training Ground Truth 类别的压缩包分别为 ISIC2018_Task1-2_Training_Input.zip 和 ISIC2018_Task1_Training_GroundTruth.zip。

    解压后，数据集目录结构如下所示：
    ```
    ├── dataset
    │   ├── ISIC2018_Task1-2_Training_Input
    │   │   ├── ISIC_<image_id>.jpg
    │   │   ├── ISIC_<image_id>.jpg
    │   │   ├── ISIC_<image_id>.jpg
    │   │   ├── ...
    │   │
    │   ├── ISIC2018_Task1_Training_GroundTruth
    │   │   ├── ISIC_<image_id>_segmentation.png
    │   │   ├── ISIC_<image_id>_segmentation.png
    │   │   ├── ISIC_<image_id>_segmentation.png
    │   │   ├── ...

    ```

2. 数据预处理。

    下载完成并解压，修改路径参数后运行dataset.py将数据集划分为三部分，分别用于training, validation, 和 test, 三部分的比例是70%, 10% 和 20%。数据集总共包含2594张图片， 其中1815张图片用于training, 259张图片用于validation，剩下的520张图片用于testing。
   
    ```
    cd /${模型文件夹名称}
    python3 dataset.py --origin_data_path=${origin_train_path} --origin_GT_path=${origin_GT_path} --train_path=${train_path} --train_GT_path=${train_GT_path} --valid_path=${valid_path} --valid_GT_path=${valid_GT_path} --test_path=${test_path} --test_GT_path=${test_GT_path} --train_ratio=0.7 --valid_ratio=0.1 --test_ratio=0.2
    ```
    参数说明：

    ```
    --origin_data_path      //原始数据集路径
    --origin_GT_path        //原始GroundTruth路径
    --train_path            //生成训练数据路径
    --train_GT_path         //生成训练GroundTruth路径
    --valid_path            //生成验证数据路径
    --valid_GT_path         //生成验证数据GroundTruth路径
    --test_path             //生成测试数据路径
    --test_GT_path          //生成测试GroundTruth路径
    --train_ratio           //训练集占比
    --valid_ratio           //验证集占比
    --test_ratio            //测试集占比
    ```
    
    处理后数据集目录结构（仅供参考，以具体路径为准）：
    ```
    ├── dataset
      ├──train
      │    │──图片1
      │    │──图片2
      │    │   ...       
      ├──train_GT
      │    │──图片1
      │    │──图片2
      │    ├──...                     
      ├──valid  
      │    │──图片1
      │    │──图片2
      │    ├──...  
      ├──valid_GT
      │    │──图片1
      │    │──图片2
      │    ├──... 
      ├──test  
      │    │──图片1
      │    │──图片2
      │    ├──...  
      ├──test_GT
      │    │──图片1
      │    │──图片2
      │    ├──...
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
        bash ./test/train_full_1p.sh --data_path=${data_path}  # 精度训练

        bash ./test/train_performance_1p.sh --data_path=${data_path}  # 性能训练
        ```

    - 单机8卡训练

        启动8卡训练。
        ```
        bash ./test/train_full_8p.sh --data_path=${data_path}  # 精度训练

        bash ./test/train_performance_8p.sh --data_path=${data_path} # 性能训练
        ```

    模型训练脚本参数说明如下。
    ```
    公共参数：
    --epoch                            //训练迭代次数
    --device_id                        //设备ID
    ```
    日志保存如下路径。
    ```
    test/output/devie_id/train_${device_id}.log          # 训练脚本原生日志
    test/output/devie_id/R2U_Net_${batch_size}_perf.log  # 8p性能训练结果日志
    test/output/devie_id/R2U_Net_${batch_size}_acc.log   # 8p精度训练结果日志
    ```
    训练模型：训练生成的模型默认会写入到和test文件同一目录下。当训练正常结束时，checkpoint.pth.tar为最终结果。

# 训练结果展示

**表 2**  训练结果展示表

| DEVICE | FPS  | Npu_nums | Epochs | BatchSize | AMP  | ACC  | Torch |
| ------ | ---- | -------- | ------ | --------- | ---- | ---- | ----- |
| V100   | 35   | 1        | 150    | 16        | O2   | 0.88 | 1.5   |
| V100   | 304  | 8        | 150    | 16*8      | O2   | 0.88 | 1.5   |
| NPU910 | 45   | 1        | 150    | 16        | O2   | 0.90 | 1.5   |
| NPU910 | 369  | 8        | 150    | 16*8      | O2   | 0.89 | 1.5   |
| NPU910 | 74   | 1        | 150    | 16        | O2   | 0.90 | 1.8   |
| NPU910 | 370  | 8        | 150    | 16*8      | O2   | 0.93 | 1.8   |

# 版本说明

## 变更

2022.07.13：更新内容，重新发布。

2020.07.08：首次发布。

## 已知问题

无。