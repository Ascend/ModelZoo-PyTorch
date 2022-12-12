# DeeplabV3 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

DeepLabV3是一个经典的语义分割网络，采用空洞卷积来代替池化解决分辨率的下降（由下采样导致），采用ASPP模型实现多尺度特征图融合，提出了更通用的框架，适用于更多网络。

- 参考实现：

  ```
  url=https://github.com/fregu856/deeplabv3
  commit_id=415d983ec8a3e4ab6977b316d8f553371a415739
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/semantic_segmentation/DeeplabV3_for_Pytorch
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
  | Python       | 3.7.5 |
  | PyTorch    | NPU版本 |
  | apex    | NPU版本 |
  | mmcv-full  | 1.3.9 |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖（根据模型需求，按需添加所需依赖）。

  ```
  bash env_set.sh
  ```

- 替换mmcv_need中的代码到mmcv-full
  ```
  mmcv_path=mmcv安装路径
  ```
  ```
  cd ${code_path}

  /bin/cp -f mmcv_need/_functions.py ${mmcv_path}/mmcv/parallel/
  /bin/cp -f mmcv_need/scatter_gather.py ${mmcv_path}/mmcv/parallel/
  /bin/cp -f mmcv_need/dist_utils.py ${mmcv_path}/mmcv/runner/
  ```
## 准备数据集

1. 获取数据集。

- 下载cityscapes数据集

- 新建文件夹data

- 将cityscas数据集放于data目录下

   ```shell
   ln -s /path/to/cityscapes/ ./data
   ```
- 配置数据集路径

  ```
  vim configs/_base_/datasets/cityscapes.py
  ```
  修改19行data_root为data文件夹路径

2. 数据预处理。

- 处理数据集，`**labelTrainIds.png` 被用来训练

   ```shell
   python3 tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
   # python3 tools/convert_datasets/cityscapes.py /path/to/cityscapes --nproc 8
   ```

## 获取预训练模型

若无法自动下载，可手动下载resnet50_v1c.pth，并放到/root/.cache/torch/checkpoints/文件夹下。

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
     # training 1p accuracy
     bash ./test/train_full_1p.sh --data_path=real_data_path  
     # training 1p performance
     bash ./test/train_performance_1p.sh --data_path=real_data_path 
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     # training 8p accuracy
     bash ./test/train_full_8p.sh --data_path=real_data_path

     # training 8p performance
     bash ./test/train_performance_8p.sh --data_path=real_data_path
     ```

   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data                              //数据集路径
   --addr                              //主机地址
   --arch                              //使用模型，默认：densenet121
   --workers                           //加载数据进程数      
   --epoch                             //重复训练次数
   --batch-size                        //训练批次大小
   --lr                                //初始学习率，默认：0.01
   --momentum                          //动量，默认：0.9
   --weight_decay                      //权重衰减，默认：0.0001
   --amp                               //是否使用混合精度
   --loss-scale                        //混合精度lossscale大小
   --opt-level                         //混合精度类型
   多卡训练参数：
   --multiprocessing-distributed       //是否使用多卡训练
   --device-list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
   ```
   
   训练完成后，权重文件保存在当前路径的output文件夹下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 |  FPS | Epochs | AMP_Type |
| ------- | ----- | ---: | ------ | -------: |
| 1p-torch1.5 | 90.97   |  6.657 | 1000     |        - |
| 1p-torch1.8  | 91.11     |  4.692 | 1000      |       O2 |
| 8p-torch1.5 | 94.49 | 36.689  | 1000    |        - |
| 8p-torch1.8  | 96.13 | 42.187 | 1000    |       O2 |

# 版本说明

## 变更

2022.08.31：更新内容，重新发布。

2020.07.08：首次发布。

## 已知问题

**_当前发行版本中存在的问题描述。_**

无。











