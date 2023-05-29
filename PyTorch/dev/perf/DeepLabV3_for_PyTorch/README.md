# DeepLabV3 for PyTorch
-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

DeepLabv3是在DeepLabv1、DeepLabv2的基础上发展而来，DeepLab系列主要围绕空洞卷积、全连接条件随机场（Fully-connected Conditional Random Field (CRF)）以及ASPP展开讨论。DeepLabv3重新考虑了空洞卷积的使用，增加了多尺度分割物体的模块，同时改进了ASPP模块。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmsegmentation.git
  commit_id=e64548fda0221ad708f5da29dc907e51a644c345
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/dev/perf/DeepLabV3_for_PyTorch
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表
  
  | Torch_Version |      三方库依赖版本      |
  | :-----------: | :----------------------: |
  |  PyTorch 1.5  | torchvision==0.2.2.post3 |
  |  PyTorch 1.8  |    torchvision==0.9.1    |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  1、使用 NPU 设备源码编译安装 mmcv-full。
  
  - 创建MMCV目录，拉取MMCV 源码
  
  ```
  git clone https://github.com/open-mmlab/mmcv.git
  ```
  
  - 编译
  
  ```
  MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python3 setup.py build_ext
  ```
  
  - 安装
  
  ```
  MMCV_WITH_OPS=1 FORCE_NPU=1 python3 setup.py develop
  ```

  2、安装 MMEngine。
  
  ```
  pip3 install mmengine==0.7.3
  ```

  3、安装 MMSegmentation
  
  下载MMSegmentation源码包:
  ```
  git clone -b main https://github.com/open-mmlab/mmsegmentation.git
  ```
  进入MMSegmentation目录，执行以下语句：
  ```
  git checkout e64548fda0221ad708f5da29dc907e51a644c345
  pip3 install -e .
  ```



## 准备数据集

1. 获取数据集。

   请用户自行获取原始数据集**cityscapes**，并在`DeepLabV3_for_PyTorch`目录新建`data`文件夹，将下载好的数据集上传至该目录`DeepLabV3_for_PyTorch/data`。

   数据集目录结构参考如下所示。

   ```
   $ DeepLabV3_for_PyTorch
       ├── data
       │   └── cityscapes
       │       ├── leftImg8bit
       │       │   ├── train
       │       │   ├── val
       │       │   └── test
       │       └── gtFine
       │           ├── train
       │           ├── val
       │           └── test
       └── ... 
   ```
   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。




# 开始训练

## 训练模型

1. 进入DeepLabV3_for_PyTorch目录。

   ```
   cd ./DeepLabV3_for_PyTorch
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh               # 单卡精度
          
     bash ./test/train_performance_1p.sh        # 单卡性能
     ```
     
   - 单机8卡训练

     启动8卡训练。
   
     ```
     bash ./test/train_full_8p.sh               # 8卡精度
     
     bash ./test/train_performance_8p.sh        # 8卡性能
     ```

   模型训练脚本参数说明如下:
   ```
   公共参数：
   --use_npu_fused_sgd                         // 是否使用NpuFusedSGD, 默认值false
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

| NAME    | mIoU  | FPS  | iters | AMP_Type | Torch_Version |
|---------|-------|------|-------|:---------|:--------------|
| 1p-竞品A  | -     | 13.6 | 40000 | O2       | 1.8           |
| 8p-竞品A  | 78.09 | 78.2 | 40000 | O2       | 1.8           |
| 8p-竞品A  | 77.47 | 62.4 | 40000 | O0       | 1.8           |
| 1p-NPU  | -     | 11.6 | 40000 | O2       | 1.8           |
| 8p-NPU  | 78.56 | 86.6 | 40000 | O2       | 1.8           |


# 版本说明

## 变更

2023.05.15: 更新训练结果。
2023.05.05: 首次发布。

## FAQ

无。
