# BiseNetV1 for Pytorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

模型基于“mmsegmentation”框架，实现了在“CityScapes”数据集上进行训练。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmsegmentation
  ```
  
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/semantic_segmentation
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version |             三方库依赖版本              |
  | :-----------: | :-------------------------------------: |
  |  PyTorch 1.5  | torchvision==0.2.2.post3；pillow==8.4.0 |
  |  PyTorch 1.8  |    torchvision==0.9.1；pillow==9.3.0    |
  
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
  
  在模型源码包根目录下执行命令，安装模型mmcv。
  
  ```
  git clone -b v1.6.1 --depth=1 https://github.com/open-mmlab/mmcv.git
  
  cp -f mmcv_need/builder.py ${mmcv_path}/mmcv/runner/optimizer/
  cp -f mmcv_need/dist_utils.py ${mmcv_path}/mmcv/runner/
  cp -f mmcv_need/__init__utils.py ${mmcv_path}/mmcv/utils/__init__.py
  cp -f mmcv_need/device_type.py ${mmcv_path}/mmcv/utils/
  cp -f mmcv_need/optimizer.py ${mmcv_path}/mmcv/runner/hooks/
  cp -f mmcv_need/__init__device.py ${mmcv_path}/mmcv/device/__init__.py
  cp -rf mmcv_need/npu ${mmcv_path}/mmcv/device/
  cp -f mmcv_need/utils.py ${mmcv_path}/mmcv/device/
  
  cd $BiseNetV1_for_PyTorch/mmcv
  pip install -r requirements/optional.txt
  MMCV_WITH_OPS=1
  FORCE_NPU=1
  python setup.py install
  ```

- 安装 mmsegmentation
  ```
  cd $BiseNetV1_for_PyTorch
  pip install -v -e .
  ```


## 准备数据集

1. 获取数据集。

   请用户自行获取原始数据集**cityscapes**，下载 `gtFine_trainvaltest.zip` (241MB) 和 `leftImg8bit_trainvaltest.zip` (11GB)。并在源码包根目录新建`data`文件夹，将下载好的数据集上传至该目录`$BiseNetV1_for_PyTorch/data`，执行以下命令进行解压。
   
   ```
   cd $BiseNetV1_for_PyTorch/data
   apt install unzip
   unzip leftImg8bit_trainvaltest.zip -d cityscapes
   # when prompting whether to overwrite, enter 'A'
   unzip gtFine_trainvaltest.zip -d cityscapes
   ```
   
   数据集目录结构参考如下所示。

   ```
   $ BiseNetV1_for_PyTorch
     ├── mmseg
     ├── tools
     ├── configs
     ├── data
     │   └── cityscapes
     │       ├── leftImg8bit
     │       │   ├── train
     │       │   ├── val
     │       │   └── test
     │       ├── gtFine
     │       │   ├── train
     │       │   ├── val
     │       │   └── test
     │       ├── train.txt
     │       ├── val.txt
     │       └── test.txt
     └── ... 
   ```
   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理（按需处理所需要的数据集）。
   
   `labelTrainIds.png`用于城市景观训练，请执行以下命令对数据进行预处理。
   
   ```
   cd $BiseNetV1_for_PyTorch
   pip install cityscapesscripts
   python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8  # --nproc表示8个转换过程，也可省略。
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
     bash ./test/train_performance_1p.sh  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh  # 8卡精度
     
     bash ./test/train_performance_8p.sh  # 8卡性能
     ```

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --load-from                         //加载权重
   --opt-level                         //混合精度类型
   --diff_seed                         //是否在不同rank上设置不同的随机数种子
   --gpu-id                            //训练卡id设置     
   --no-validate                       //设置训练期间是否评测
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| Name      | mIoU  | FPS |  Device  | Npu_nums | Steps | AMP_Type | CPU |
|-----------|:-----:|:---:|:--------:|:--------:|:-----:|:--------:|:---:|
| 1p-*PU    |   -   |  9  |    -     |    -     |  400  |    O1    | x86 |
| 1p-NPU1.8 |   -   | 12  |   910A   |    1     |  400  |    O1    | ARM |
| 8p-*PU    | 75.80 | 62  |    -     |    -     | 40000 |    O1    | x86 |
| 8p-NPU1.8 |   -   | 88  |   910A   |    8     |  400  |    O1    | ARM |
| 8p-NPU1.8 | 76.03 | 88  |   910A   |    8     | 40000 |    O1    | ARM |

# 版本说明

## 变更

2023.03.06：更新readme，重新发布。

2020.07.08：首次发布。

## FAQ

建议使用“python”或“python3.7”来执行模型训练过程。如果需要使用“python3”，请在使用“python3”之前运行以下命令。

```
unlink /usr/bin/python3
ln -s /usr/bin/python3.7 /usr/bin/python3
```
