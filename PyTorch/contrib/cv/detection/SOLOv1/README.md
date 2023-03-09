# SOLOv1 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

SOLOv1框架的核心思想是按位置分割对象。输入图像在概念上分为S×S网格。如果对象的中心落在网格单元中，则该网格单元负责预测语义类别以及分配每像素位置类别。有两个分支：类别分支和掩码分支。类别分支预测语义类别，而掩码分支分割对象实例。SOLO实现了端到端的训练，无需anchor，通过离散量化，将坐标回归转化为分类问题，可以避免启发式的坐标规范问题，简化了实例分割任务，具有较好的性能，且实现了和Mask-R-CNN基本持平的效果。

- 参考实现：

  ```
  url=https://github.com/WXinlong/SOLO
  commit_id=95f3732d5fbb0d7c7044c7dd074f439d48a72ce5
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
  | PyTorch 1.5 | torchvision==0.2.2.post3；pillow==8.4.0 |
  | PyTorch 1.8 | torchvision==0.9.1；pillow==9.1.0 |
  
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

- 安装mmcv。
  
  ```
  cd mmcv
  source test/env_npu.sh  
  python3.7 setup.py build_ext
  python3.7 setup.py develop
  cd ..
  pip3 list | grep mmcv  # 查看版本和路径
  ```

- 安装mmdet。

   ```
  pip install -r requirements/build.txt
  pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
  pip install -v -e .
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集**coco2017**，将coco数据集放于`SOLOv1/data`目录下，数据集目录结构参考如下所示。

   ```
   SOLOv1
     ├── configs
     ├── data
     │   ├── coco
     │       ├── annotations   796M
     │       ├── train2017     19G
     │       ├── val2017       788M            
   ```
   
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。


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
     bash ./test/train_full_1p.sh --data_path=./data/coco  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=./data/coco  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=./data/coco  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=./data/coco  # 8卡性能
     ```

   - 单机单卡评测

     启动单卡评测。

     ```
     bash ./test/train_eval_1p.sh --data_path=./data/coco
     ```

   - 多机多卡性能

     启动多机多卡性能训练。

     ```
     1. 安装环境
     2. 开始训练，每个机器请按下面提示进行配置
     bash ./test/train_performance_multinodes.sh --data_path=数据集路径 --batch_size=单卡batch_size*所有卡数 --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --opt-level                         //混合精度类型
   --seed                              //随机数种子设置
   --addr                              //主机地址
   --data_root                         //数据集路径  
   --validate                          //设置在训练期间其否评测
   --total_epochs                      //训练周期数
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME | Acc@1    | FPS       | Epochs   | AMP_Type |
| :------: | :------:  | :------:     | :-----: | :-----: |
| 1p-竞品V | - | 2.75      | 1        | O1    |
| 8p-竞品V | 32.3     | 16.3      | 12       | O1    |
| 1p-NPU | - | 1.42      | 1        | O1    |
| 8p-NPU | 32.1     | 9.4       | 12       | O1    |

# 版本说明

## 变更

2023.03.10：更新readme，重新发布。

2020.07.08：首次发布。

## FAQ

无。
