# FCN8s for PyTorch\_Owner

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述
FCN8s是一个经典的语义分割网络，FCN8s使用全卷积结构，可以接受任意尺寸的输入图像，采用反卷积对最后一层的特征图进行上采样，得到与输入图像相同尺寸的输出，从而对输入进行逐像素预测。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmsegmentation.git
  commit_id=9f071cade8cdc59c13b416c7c9843005410c055c
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/semantic_segmentation
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

- 构建mmcv。

  下载[mmcv1.2.7](https://github.com/open-mmlab/mmcv/tree/v1.2.7)到路径`$YOURMMVCPATH`。然后，复制`mmcv_replace` 到 `$YOURMMVCPATH/mmcv`。
  ```
  # configure
  cd /${模型文件夹名称}
  source env_npu.sh

  # copy
  rm -rf $YOURMMVCPATH/mmcv
  mkdir mmcv
  cp -r mmcv_replace/* $YOURMMVCPATH/mmcv/

  # compile
  cd $YOURMMVCPATH
  export MMCV_WITH_OPS=1
  export MAX_JOBS=8
  python3 setup.py build_ext
  python3 setup.py develop
  pip3.7.5 list | grep mmcv
  
  cd /${模型文件夹名称}
  ```

- 权限配置。
  ```
  chmod -R 777 ./
  ```
  
- 删除 `mmcv_replace` 文件夹。
  ```
  rm -rf mmcv_replace
  ```

## 准备数据集

1. 获取数据集。

  - 下载 PASCAL VOC 2012 数据集和 PASCAL VOC2010 数据集的训练集和验证集。

    解压后，数据集目录结构如下所示：

    ```none
    ├── VOCdevkit
    │   │   ├── VOC2012
    │   │   ├── VOC2010
    ```


  - 下载 [PASCALAug](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)数据集。
  
    解压后，复制`benchmark_REALSE/dataset`到`VOCdevkit`文件夹下的`VOCaug`。

    数据集目录结构如下所示：
    ```none
    ├── VOCdevkit
    │   │   ├── VOC2012
    │   │   ├── VOC2010
    │   │   ├── VOCaug
    ```
2. 数据预处理。
- 使用以下命令转换 VOCAug 数据集。

  ```
  cd /${模型文件夹名称}
  python tools/convert_datasets/voc_aug.py data/VOCdevkit data/VOCdevkit/VOCaug --nproc 8
  ```
  >**说明:** 可能会提示`Segmentation fault (core dumped)` ， 提示原因是mmcv需要pytorch支持。 请返回源码包根目录并运行 `source env_npu.sh`。 

- [可选] 建立数据集到文件夹mmseg100的软链。
  ```
  cd /${模型文件夹名称}
  mkdir data
  ln -s VOCdevkit data # data_path=./data/VOCdevkit/VOC2012
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
     # training 1p accuracy
     bash ./test/train_full_1p.sh --data_path=xxx --device_id=xxx
     # --data_path=data/VOCdevkit/VOC2012
     # --device_id 指定训练用卡，可选0，1，2，3，4，5，6，7

     # training 1p performance
     bash ./test/train_performance_1p.sh --data_path=xxx --device_id=xxx
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     # training 8p accuracy
     bash ./test/train_full_8p.sh --data_path=xxx 

     # training 8p performance
     bash ./test/train_performance_8p.sh --data_path=xxx 

     # evaluation 8p accuracy
     bash ./test/train_val_8p.sh --data_path=xxx 
     ```

   - 多机多卡性能数据获取流程

     ```shell
     	1. 安装环境
     	2. 开始训练，每个机器所请按下面提示进行配置
             bash ./test/train_performance_multinodes.sh  --data_path=数据集路径 --batch_size=单卡batch_size --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
     ```

    --data\_path参数填写数据集路径。

  模型训练脚本参数说明如下。
  ```
  公共参数：
  --device                            //指定gpu或npu
  --data_path                         //数据集路径
  --addr                              //主机地址     
  --amp                               //是否使用混合精度
  --loss-scale                        //混合精度lossscale大小
  --opt-level                         //混合精度类型
  多卡训练参数：
  --device-list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
  ```

  日志和权重文件保存在如下路径。
  ```
  ./output/devie_id/FCN/train_${device_id}.log          # training detail log
  ./output/devie_id/FCN/FCN_bs16_8p_acc.log             # 8p training performance result log
  ./output/devie_id/FCN/ckpt                            # checkpoits
  ./output/devie_id/FCN_prof/FCN_bs16_8p_acc.log        # 8p training accuracy result log
  ```

# 训练结果展示

**表 2**  训练结果展示表

| 名称    |  FPS   |  aAcc |  mIoU |  mAcc |
| :------: | :------: | :------: | :------: | :------: |
| 1p-竞品 | -----  | ----- | ----- | ----- |
| 1p-NPU  | 28.46 | 90.73 | 59.61 | 72.01 |
| 8p-竞品 | 135.19 | 93.23 | 69.36 | 78.88 |
| 8p-NPU  | 193.15 | 93.06 | 68.41 | 77.39 |


# 版本说明

## 变更

2022.09.06：更新内容，重新发布。

2020.07.08：首次发布。

## 已知问题

无。


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md







