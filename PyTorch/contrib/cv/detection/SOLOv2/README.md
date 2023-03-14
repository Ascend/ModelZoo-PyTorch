# SOLOV2 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述
SOLOv2建立了一个简单，直接，快速的实例分割框架，具有很强的性能。遵循SOLO的设计原则。即按位置分割对象。进一步通过动态学习对象分段器的mask头，使得mask头受位置的约束。具体地，将掩模分支分解为掩模核分支和掩模特征分支，分别负责卷积核和卷积特征的学习。此外，提出矩阵NMS（非最大抑制）来显著减少由于NMSofmasks引起的推理时间开销。matrix NMS一次完成了并行矩阵运算的NMS，并获得了更好的结果。一个简单的直接实例分割系统，在速度和精度上都优于一些最新的方法。一个轻量级版本的SOLOv2以31.3fps的速度运行，产生37.1%的AP。

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

- 配置环境。
   ```
   # 进入SOLOv2目录，source环境变量
   cd SOLOv2
   source test/env_npu.sh  
   ```

- 安装 `mmcv`。

   ```
   cd mmcv
   python3.7 setup.py build_ext
   python3.7 setup.py develop
   cd ..
   pip list | grep mmcv  # 查看版本和路径
   ```
- 安装 `mmdet`。

   ```
   pip install -r requirements/build.txt
   pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
   pip install -v -e .
   ```
   
```
注意：安装mmdet的时候，如果自动卸载已安装torch，可以使用命令pip install -v -e . --no-deps安装
```
## 准备数据集

1. 获取数据集。

   用户自行下载 `coco2017` 数据集，在模型根目录下新建 `data` 文件夹，并将 `coco` 数据集放于 `data` 目录下，数据集目录结构参考如下所示。
  
   ```
    SOLOV2
      ├── configs
      ├── data
            ├── coco
                ├── annotations 
                ├── train2017   
                ├── val2017 
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

## 获取预训练模型

用户自行下载 `resnet50-19c8e357.pth` 预训练权重文件，下载完成后重命名为 `resnet50.pth`，并放至在 `data` 目录下新建的 `pretrained` 目录下。最终目录结构参考如下所示。
   
 ```
 SOLOV2
   ├── configs
   ├── data
         ├── coco
             ├── annotations 
             ├── train2017   
             ├── val2017     
         ├── pretrained
             ├── resnet50.pth
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
     bash ./test/train_full_1p.sh --data_path=./data/coco # 单卡精度
     bash ./test/train_performance_1p.sh --data_path=./data/coco  # 单卡性能
     ```

   - 单机8卡训练
     
     启动8卡训练。

      ```
      bash ./test/train_full_8p.sh --data_path=./data/coco  # 8卡精度
      bash ./test/train_performance_8p.sh --data_path=./data/coco # 8卡性能
      ```

   - 单机单卡评测
     
     启动单机单卡评测。

     ```
     bash ./test/train_eval_1p.sh --data_path=./data/coco # 单卡评测
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。
  
   模型训练脚本参数说明如下。

   ```
   公共参数：
   --opt-level                        //apex级别
   --autoscale-lr                     //是否自动缩放lr通过gpu的数量
   --seed                             //随机数种子
   --total_epochs                     //训练轮数
   --data_root                        //数据路径
   --gpu-ids                          //设备id
   --train_performance                //performance模式 1/ full模式 0
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   Name   | Acc@1    | FPS       | Epochs   | AMP_Type | Torch_Version |
| :------:     | :------: | :------:  | :------: | :------: | :------:  |
| 1p-竞品V     | 18.4     | 2.5       | 1        | O1       | 1.5      |
| 8p-竞品V     | 34.4     | 14.9      | 12       | O1       | 1.5      |
| 1p-Npu       | 18.4     | 2.45       | 1        | O1       | 1.8      |
| 8p-Npu       | 34.3     | 15.34       | 12       | O1       | 1.8      |


# 版本说明

## 变更

2022.4.17：首次发布。

## FAQ

无。








