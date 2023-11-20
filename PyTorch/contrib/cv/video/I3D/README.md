# I3D for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述
I3D是一种新型的基于二维卷积网络膨胀生成的双流三位卷积网络。它将卷积网络内的非常深的图像分类的过滤器和池化内核扩展到三维，使得从视频中学习无缝时空特征提取器成为可能，同时利用了成功的ImageNet架构及其参数，从而可以实现非常好的视频动作特征识别效果。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/i3d
  commit_id=ace8beb46399bd0c881cdeccbfcb0ed1fa7b31ad
  ```
  
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/video
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version |   三方库依赖版本   |
  | :-----------: | :----------------: |
  |  PyTorch 1.5  | torchvision==0.6.0 |
  |  PyTorch 1.8  | torchvision==0.9.1 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  ```
  > **说明：** 
  > 只需执行一条对应的PyTorch版本依赖安装命令。

- 安装 MMCV。

  在源码包根目录下执行以下命令，下载三方库源码，在本地进行编译安装。
  ```
  # 拉取源码包
  git clone -b v1.3.9 https://github.com/open-mmlab/mmcv.git
  
  # 文件替换
  cp -f mmcv_need/base_runner.py mmcv/runner/base_runner.py
  cp -f mmcv_need/builder.py mmcv/runner/optimizer/builder.py
  cp -f mmcv_need/checkpoint.py mmcv/runner/hooks/checkpoint.py
  cp -f mmcv_need/data_parallel.py mmcv/parallel/data_parallel.py
  cp -f mmcv_need/dist_utils.py mmcv/runner/dist_utils.py
  cp -f mmcv_need/distributed.py mmcv/parallel/distributed.py
  cp -f mmcv_need/epoch_based_runner.py mmcv/runner/epoch_based_runner.py
  cp -f mmcv_need/iter_timer.py mmcv/runner/hooks/iter_timer.py
  cp -f mmcv_need/optimizer.py mmcv/runner/hooks/optimizer.py
  cp -f mmcv_need/test.py mmcv/engine/test.py
  cp -f mmcv_need/transformer.py mmcv/cnn/bricks/transformer.py
  cp -f mmcv_need/registry.py mmcv/utils/registry.py
  
  # 编译
  source test/env_npu.sh
  cd mmcv
  export MMCV_WITH_OPS=1 
  export MAX_JOBS=8
  python3 setup.py build_ext
  python3 setup.py develop
  pip3.7 list | grep mmcv
  ```
  
- 安装mmaction2
  
   安装方式可参考源码readme介绍。
   
   ```
   https://github.com/open-mmlab/mmaction2/blob/master/docs_zh_CN/install.md
   
   # 注意安装不含cuda算子的版本, 同时将本模型目录里的“mmaction”完全替换三方库源码“mmaction2/mmaction”目录文件。
   ```


## 准备数据集

1. 获取数据集。

   请用户自行下载**kinetics400**数据集，包含训练集和验证集两部分，训练集和验证集图片分别位于“train/”和“val/”文件夹路径下，该目录下每个文件夹代表一个类别，且同一文件夹下的所有图片都有相同的标签，将下载好的数据集上传到服务器任意路径下并解压。

   数据集目录结构参考如下所示。

   ```
   ├── kinetics400
   │    ├──train├──类别1──视频1、2、3、4
   │    │       │
   │    │       ├──类别2──视频1、2、3、4
   │    │
   │    ├──val  ├──类别1──视频1、2、3、4 
   │  		     │ 
   │		     ├──类别2──视频1、2、3、4
   │    ├──train_label
   │    ├──val_label
   ```
   
   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。


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

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh --data_path=/data/xxx/
     ```

   --data_path参数填写数据集路径，需写到数据集的上一级目录（示例：数据集地址为：/home/dataset/kinetics400， 则`--data_path=/home/dataset`）。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --config                            //训练配置文件路径
   --seed                              //随机数种子设置
   --data_root                         //数据集路径
   --cfg-options                       //参数配置
   --work-dir                          //日志和模型保存目录
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Top1 acc |  FPS  | Epochs | AMP_Type |
| :------: | :------: | :---: | :----: | :------: |
| 1p-竞品V |    -     | 20.68 |   1    |    O1    |
| 8p-竞品V |  53.78   | 74.84 |   40   |    O1    |
|  1p-NPU  |    -     | 18.41 |   1    |    O1    |
|  8p-NPU  |  56.26   | 58.92 |   40   |    O1    |

# 版本说明

## 变更

2023.03.21：更新readme，重新发布。

2022.10.17：首次发布。

## FAQ

1. 在安装decord库的时候apt_pak报错

   ```
   使用如下命令查看python版本
   ls /usr/lib/python3/dist-packages/apt_pkg*
   使用下面命令：
   vim /usr/bin/apt-add-repository
   把首行的
   #! /usr/bin/python3
   改为对应版本（我这里是python3.6）
   #! /usr/bin/python3.6
   ```

2. 在ARM平台上，安装0.6.0版本的torchvision，需进行源码编译安装，可以参考源码readme进行安装。
   
   ```
   https://github.com/pytorch/vision
   ```

3. 在ARM平台上，安装decord，需进行源码编译安装，可以参考源码readme进行安装。
   
   ```
   https://github.com/dmlc/decord/blob/master/README.md
   ```

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md   