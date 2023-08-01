# CenterFace for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述
Centerface的非官方版本，实现了速度和准确性之间的最佳平衡。Centerface是一种适用于边缘设备的实用无锚目标检测和对齐方法。项目提供训练脚本、训练数据集和预训练模型，方便用户复现结果。

- 参考实现：

  ```
  url=https://github.com/chenjun2hao/CenterFace.pytorch
  commit_id=585ce2c4fb80ae6ab236f79f06911e2f8bef180c
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
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |
  | PyTorch 1.11   | - |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。


- 安装依赖。

  ```shell
  # 安装依赖包
  # for pip
  cd ${模型文件夹名称}
  pip install -r requirements.txt
  # for conda
  conda env create -f enviroment.yaml
  ```
  ```
  # 编译（编译过的可跳过，编译需要先执行以下操作，否找可能出现 ModuleNotFoundError: No module named 'external.nms'）
  cd ${模型文件夹名称}/src/lib/external
  make
  ```
  ```
  # 安装评测时用到的bbox库扩展方法bbox_overlaps
  cd ${模型文件夹名称}/evaluate
  python3 setup.py install
  ```  
## 准备数据集
   * 准备数据

     请用户自行准备好数据集，包含训练集和验证集两部分，可选用的数据集包括WIDER-FACE等。上传到服务器任意路径并解压；“WIDER_FACE_DATA_ALL.zip”文件里面有“annotations.zip”、“labels”、“WIDER_train.zip”、“WIDER_val.zip”、“groud_truth”文件。

     在当前源码包根目录下建立“data/wider_face/image”文件夹。将“annotations.zip”、“labels”、“WIDER_train.zip”、“WIDER_val.zip”复制到服务器的源码包根目录“data/wider_face”目录下并解压，“groud_truth”复制到源码包根目录下。将“WIDER_train”中的“images”复制到源码包根目录下的“data/wider_face/image”文件夹中。数据集目录结构参考：
     ```
     ├── data
         ├──wider_face
             ├──labels
             ├──WIDER_train
             ├──WIDER_val
             ├──annotations
             ├──image
                   ├──0-Parade		分类1
                         ├──图片1
                         ├──图片2
                         ...
                   ├──1-Handshaking	分类2
                         ├──图片1
                         ├──图片2
                         ...
                         ..
     ├── groud_truth
          ├── wider_easy_val.mat
          ├── wider_face_val.mat
          ├── wider_hard_val.mat
          ├── wider_medium_val.mat
     ```
     > **说明：**
     >该数据集的训练过程脚本只作为一种参考示例。
## 获取预训练模型

请参考原始仓库上的README.md进行预训练模型获取。


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
      bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度

      bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
      ```
   - 单机8卡训练

     启动8卡训练。
      ```
      bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度

      bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
      ```

    --data_path参数填写数据集路径，需写到数据集的一级目录。

    模型训练脚本参数说明如下。

    ```
    公共参数：
    --batch_size                          // 数据批大小
    --device-list                         // 设备id
    --world_size                          // 使用卡数量
    --lr                                  // 学习率
    --lr_step                             // 多少个step学习率调整
    --port                                // 分布式master端口
    --num_epochs                          // 训练轮数
    ```

   训练日志路径：网络脚本test下output文件夹内。例如：

    ```
    test/output/devie_id/CenterFace_${device_id}.log  # 训练脚本原生日志

    test/output/devie_id/CenterFace_bs1024_8p_perf.log  # 8p性能训练结果日志

    test/output/devie_id/CenterFace_bs1024_8p_acc.log   # 8p精度训练结果日志
    ```
    训练完成后，训练生成的模型默认会写入到和test文件同一目录下。当训练正常结束时，checkpoint.pth.tar为最终结果。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Acc | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V |   -   | 28  |   1    |    -     |      1.5      |
| 8p-竞品V | easy:85.75;medium:84.96;hard:67.15 | 235 |  140   |    -     |      1.5      |
|  1p-NPU  |   -   | 38.73  |   1    |    O1    |      1.8      |
|  8p-NPU  |  easy:87.03;medium:86.50;hard:70.17  | 281.65 |  140   |    O1    |      1.8      |


# 版本说明

## 变更

2021.10.17：首次发布。

## 已知问题

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md


