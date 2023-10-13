# RetinaNet_for_PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

RetinaNet提出了一种使用Focal Loss的全新结构RetinaNet，使用ResNet+FPN作为backbone，再利用单级的目标识别法+Focal Loss。本文档描述的Retinanet是基于Pytorch实现的版本。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection/tree/v2.9.0/configs/retinanet
  commit_id=6c1347d7c0fa220a7be99cb19d1a9e8b6cbf7544
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/detection
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.2.2.post3 |
  | PyTorch 1.8 | torchvision==0.9.1 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。
- 编译安装mmcv。

  ```
  cd ${模型文件夹名称} （源码包根目录）
  git clone -b v1.2.6 --depth=1 https://github.com/open-mmlab/mmcv.git

  export MMCV_WITH_OPS=1
  export MAX_JOBS=8
  source test/env_npu.sh

  cd mmcv
  python3 setup.py build_ext
  python3 setup.py develop
  pip3.7 list | grep mmcv
  ```

- 安装mmdet。

  ```
  cd RetinaNet_for_PyTorch
  pip3.7 install -r requirements/build.txt
  pip3.7 install -v -e .
  pip3.7 list | grep mmdet
  ```

- 替换mmcv部分文件。

  ```
  cd ${模型文件夹名称}/mmcv_need
  cp -f ./_functions.py ../mmcv/mmcv/parallel/
  cp -f ./builder.py ../mmcv/mmcv/runner/optimizer/
  cp -f ./data_parallel.py ../mmcv/mmcv/parallel/
  cp -f ./dist_utils.py ../mmcv/mmcv/runner/
  cp -f ./distributed.py ../mmcv/mmcv/parallel/
  cp -f ./optimizer.py ../mmcv/mmcv/runner/hooks/ 
  ```

## 准备数据集

1. 下载COCO数据集

2. 在源码包根目录下新建"data"文件夹

3. 将coco数据集解压后放至于"data"目录下

   以coco2017数据集为例，数据集目录结构参考如下所示。

   ```
   ├── coco2017
   │   ├── annotations
   │          ├── captions_train2017.json
   │          ├── captions_val2017.json
   │          ├── instances_train2017.json
   │          ├── instances_val2017.json
   │          ├── person_keypoints_train2017.json
   │          ├── person_keypoints_val2017.json
   │   ├── train2017
   │          ├── 000000000009.jpg
   │          ├── 000000000025.jpg
   │          ├── ......
   │   ├── val2017
   │          ├── 000000000139.jpg
   │          ├── 000000000285.jpg
   │          │   ...              
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
     chmod +x ./tools/dist_train.sh
     bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练
   
     启动8卡训练。
   
     ```
     chmod +x ./tools/dist_train.sh
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```
   
   - 多机多卡性能数据获取流程。

     ```
     bash ./test/train_performance_multinodes.sh --data_path=数据集路径 --batch_size=单卡batch_size --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
     ```
   
   --data_path参数填写数据集路径，需写到数据集的一级目录。
   
   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --opt-level                         //混合精度类型
   --addr                              //主机地址
   --seed                              //训练的随机数种子
   --gpu-ids                           //使用训练卡id
   --work-dir                          //模型保存日志
   --gpus                              //训练卡使用个数
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|  NAME  |  mAP  | FPS  | Epochs | AMP_Type | Torch_Version |
| :----: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-NPU | - |  17  |   1    |    O1    |      1.11      |
| 8p-NPU（物理机） | 0.360 | 112  |   12   |    O1    |      1.11      |
| 8p-NPU（容器） | 0.360 | 98  |   12   |    O1    |      1.11      |

* 说明：由于容器中CPU利用率较低，该模型在容器中训练较物理机会有性能劣化

# 版本说明

## 变更

2023.02.14：更新readme，重新发布。

2021.04.12：首次发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md