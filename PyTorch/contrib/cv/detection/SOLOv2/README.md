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
  | 硬件 | [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

  
- 安装依赖。

  ```
   pip install -r requirements.txt
  ```

- 配置环境。
   ```
   # 进入SOLOv2目录，source环境变量
   cd SOLOv2
   source test/env_npu.sh  
   ```

  配置安装mmcv

   ```
   cd mmcv
   python3.7 setup.py build_ext
   python3.7 setup.py develop
   cd ..
   pip list | grep mmcv  # 查看版本和路径
   ```
  配置安装mmdet

   ```
   pip install -r requirements/build.txt
   pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
   pip install -v -e .
   ```
## 准备数据集 & 预训练模型

   * 下载coco2017数据集
   * 用户自行下载resnet50-19c8e357.pth预训练权重文件，下载完成后重命名为resnet50.pth
   * 将coco数据集放于SOLOv2/data目录下，目录结构自行构建如下：

      ```
      SOLOV2
      ├── configs
      ├── data
      │   ├── coco
      │       ├── annotations 
      │       ├── train2017   
      │       ├── val2017     
      │   ├── pretrained
      │       ├── resnet50.pth
      ```

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。
   - 单机单卡训练
     ```
     # 导入环境变量，修改train_full_1p.sh权限并运行
     chmod +x ./test/train_full_1p.sh
     bash ./test/train_full_1p.sh --data_path=./data/coco
      
     # 导入环境变量，修改train_performance_1p.sh权限并运行
     chmod +x ./test/train_performance_1p.sh
     bash ./test/train_performance_1p.sh --data_path=./data/coco
     ```

   - 单机8卡训练
      ```
      # 导入环境变量，修改train_full_8p.sh权限并运行
      chmod +x ./test/train_full_8p.sh
      bash ./test/train_full_8p.sh --data_path=./data/coco
      # 导入环境变量，修改train_performance_8p.sh权限并运行
      chmod +x ./test/train_performance_8p.sh
      bash ./test/train_performance_8p.sh --data_path=./data/coco
      ```

   - 单卡评估

     修改train_eval_1p.sh权限并运行
  
     ```
     chmod +x ./test/train_eval_1p.sh
     bash ./test/train_eval_1p.sh --data_path=./data/coco
     ```

   - 单卡微调

     修改train_finetune_1p.sh权限并运行
  
     ```
     chmod +x ./test/train_eval_1p.sh
     bash ./test/train_finetune_1p.sh --data_path=./data/coco
     ```

  --data_path: 数据集路径
  
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

   日志和权重文件保存在生成output目录下

# 训练结果展示

**表 2**  训练结果展示表

| Npu/Gpu_nums | Acc@1    | FPS       | Epochs   | AMP_Type | Loss_Scale |
| :------:     | :------: | :------:  | :------: | :------: | :------:   |
| 1p Gpu       | 18.4     | 8.1       | 1        | O1       | 128.0      |
| 8p Gpu       | 34.8     | 38.3      | 12       | O1       | 128.0      |
| 1p Npu       | 18.4     | 1.1       | 1        | O1       | 128.0      |
| 8p Npu       | 34.3     | 6.2       | 12       | O1       | 128.0      |


# 版本说明

## 变更

2022.4.17：首次发布。

## 已知问题

无。








