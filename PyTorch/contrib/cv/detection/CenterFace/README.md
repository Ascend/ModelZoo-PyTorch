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
  | 硬件       | [1.0.11](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)
  | 固件与驱动 | [21.0.2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.0.2](https://www.hiascend.com/software/cann/commercial?version=5.0.2) |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/)       |

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
    # 编译（编译过的可跳过，编译需要先执行以下操作，否找可能出现报错ModuleNotFoundError: No module named 'external.nms'）
    cd ${模型文件夹名称}/src/lib/external
    make
  ```
## 准备数据集 & 预训练模型
   * 准备数据

     1，请用户自行准备好数据集，包含训练集和验证集两部分，可选用的数据集包括WIDER-FACE和模型链接https://download.pytorch.org/models/mobilenet_v2-b0353104.pth。

     2，本机解压“WIDER_FACE_DATA_ALL.zip”文件里面有“annotations.zip”、“labels”、“WIDER_train.zip”、“WIDER_val.zip”、“groud_truth”文件。在当前源码包根目录下建 
        立“data/wider_face/image”文件夹。将“annotations.zip”、“labels”、“WIDER_train.zip”、“WIDER_val.zip”复制到服务器的源码包根目录“data/wider_face”目录下 
        并解压，“groud_truth”复制到源码包根目录下。将“WIDER_train”中的“images”复制到源码包根目录下的“data/wider_face/image”文件夹中。数据集目录结构参考：
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
               |--mobilenet_v2-b0353104.pth
      ```
   * 下载预训练模型

        下载预训练模型：[link](https://pan.baidu.com/s/1sU3pRBTFebbsMDac-1HsQA)，password: etdi。



# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

    - 启动1p训练
      ```
      bash ./test/train_full_1p.sh --data_path=/data/xxx/

      bash ./test/train_performance_1p.sh --data_path=/data/xxx/
      ```
    - 启动8p训练
      ```
      bash ./test/train_full_8p.sh --data_path=/data/xxx/

      bash ./test/train_performance_8p.sh --data_path=/data/xxx/
      ```

   --data_path：数据集路径。

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

      test/output/devie_id/CenterFace_${device_id}.log          # 训练脚本原生日志

      test/output/devie_id/CenterFace_bs1024_8p_perf.log  # 8p性能训练结果日志

      test/output/devie_id/CenterFace_bs1024_8p_acc.log   # 8p精度训练结果日志

    训练模型：训练生成的模型默认会写入到和test文件同一目录下。当训练正常结束时，checkpoint.pth.tar为最终结果。

# 训练结果展示

**表 2**  训练结果展示表

|名称  | 精度  |  性能     |
|----| ----- | ---------- |
|GPU1p| -    | 28  |
|NPU1p| -    |  34.5    |
|GPU8p| easy:85.75;medium:84.96;hard:67.15 | 235    |
|NPU8p| easy:87.03;medium:86.50;hard:70.17 | 41.5    |


# 版本说明

## 变更

2021.10.17：首次发布。

## 已知问题

无。








