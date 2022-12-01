# SSD for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

SSD 模型利用不同尺度的特征图进行目标的检测，SSD 采用多个尺度检测方法是将 VGG16 网络输出的大特征图逐步采用步长为 2的卷积操作，生成不同大小的特征图。本项目实现了 SSD (Single Shot MultiBox Detector) 在 NPU 上的训练，迁移自 [MMDetection](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd)，[MMCV](https://github.com/open-mmlab/mmcv/)。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd
  commit_id=e33ecfed37f594050e13537972e65f0ccf079982c
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/detection
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
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```

- 导入环境变量。
    ```
    source test/env_npu.sh
    ```

- 安装 MMCV。
    ```
    git clone -b v1.2.7 git://github.com/open-mmlab/mmcv.git
    
    cp -f mmcv_need/_functions.py mmcv/mmcv/parallel/
    cp -f mmcv_need/builder.py mmcv/mmcv/runner/optimizer/
    cp -f mmcv_need/data_parallel.py mmcv/mmcv/parallel/
    cp -f mmcv_need/dist_utils.py mmcv/mmcv/runner/
    cp -f mmcv_need/distributed.py mmcv/mmcv/parallel/
    cp -f mmcv_need/optimizer.py mmcv/mmcv/runner/hooks/
    
    cd mmcv
    export MMCV_WITH_OPS=1 
    export MAX_JOBS=8
    python3.7 setup.py build_ext
    python3.7 setup.py develop
    pip3.7 list | grep mmcv
    cd ..
    ```

- 安装 MMDetection。
    ```
    pip3.7 install -r requirements/build.txt
    pip3.7 install -v -e .
    pip3.7 list | grep mmdet
    ```



## 准备数据集

1. 获取数据集。
    
   用户自行获取coco数据集，将数据集上传到任意路径下并解压在当前路径。
    
   coco数据集目录结构参考如下所示（当前数据集所在位置为在源码包根目录下新建的“data/”目录下）。

   ```
   SSD
   ├── configs
   ├── data
   │   └── coco
   │       ├── annotations
   │       ├── train2017
   │       ├── val2017
   │       └── test2017
   ├── mmcv
   ├── mmdet
   ├── tools
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
     bash ./test/train_full_1p.sh --data_path=xxx 
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=xxx
     ```
   - 多机多卡性能数据获取流程

     ```shell
     1. 安装环境
     2. 开始训练，每个机器所请按下面提示进行配置
             bash ./test/train_performance_multinodes.sh  --data_path=数据集路径 --batch_size=单卡batch_size --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
     ```
   --data_path参数填写解压后的数据集路径。

   模型训练脚本参数说明如下

   ```
   公共参数：
   config                              //训练配置文件路径
   --work-dir                          //保存日志和模型的目录
   --resume-from                       //检查点文件
   --no-validate                       //训练期间是否不评估检查点
   --gpus                              //训练卡的数量
   --gpu-ids                           //训练卡的ids
   --seed                              //随机种子
   --options                           //合并配置文件(不推荐)
   --cfg-options                       //合并配置文件(推荐)
   --launcher                          //作业启动器

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

 3. 计算FPS值。

    FPS 可使用 calc_fps.py 计算，使用方法为：
    ```
    python3.7 calc_fps.py xxx.log.json ${gpu_nums} ${batch_size}
    ```
    参数batch_size是模型训练加载图片的批次大小。

    参数gpu_nums是模型训练时启动的卡数。

    xxx.log.json是模型训练完成后生成的json文件，用于计算FPS值。


# 训练结果展示

**表 2**  训练结果展示表

| NAME   | 0.5:0.95mAP |  FPS | Epochs | AMP_Type | pytorch |
| ------ | -------- | -----: | ------ | -------: | ------ |
| 1p-npu | -    | 20.6(x86 cpu)  | 1     |   O1 | 1.5 |
| 1p-npu | -    | 12.6           | 1     |   O2 | 1.8 |
| 8p-npu | 25.5 | 64.5(ARM CPU)  | 24    |   O1 | 1.5 |
| 8p-npu | 25.5 | 66.3           | 24    |   O2 | 1.8 |


# 版本说明

## 变更

2022.10.14：更新torch1.8版本，重新发布。

2020.07.08：首次发布。

## 已知问题


无。

