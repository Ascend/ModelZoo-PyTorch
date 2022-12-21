# MobileNetv1 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md) 
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述
## 简述

 MobileNetv1是一种基于流水线结构，使用深度级可分离卷积构建的轻量级神经网络，并通过两个超参数的引入使得开发人员可以基于自己的应用和资源限制选择合适的模型。

- 参考实现：

  ```
  url=https://github.com/wjc852456/pytorch-mobilenet-v1
  commit-id=8b3bde3e525ba6d17b9cabb5feb8ee49a9e1e8e0
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification
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
  | 硬件 | [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)|

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```
  注意: pillow建议安装更新的版本。如果无法直接安装对应版本的torchvision，可以使用源代码安装对应版本。源代码参考链接:https://github.com/pytorch/vision， 建议pilow为9.1.0，torchvision为0.6.0

## 准备数据集

1. 获取数据集。

   下载开源数据集包括ImageNet，将数据集上传到服务器任意路径下并解压。
   
   数据集目录结构参考如下所示。

   ```
   ├── ImageNet
         ├──train
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...   
              ├──...                     
         ├──val  
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...              
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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/    
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/   
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/
     ```

    --data_path: 数据集路径

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --arch                         //模型名称
   --data                         //数据集路径
   --batch_size                   //训练批次大小
   --learning-rate                //初始学习率
   --print-freq                   //打印频率
   --epochs                       //重复训练次数
   --apex                         //使用混合精度
   --apex-opt-level               // apex优化器级别
   --loss-scale-value             // apex loss缩放比例值
   --device_id                    //使用设备
   --workers                      //加载数据线程数
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1  |    FPS  | Epochs | AMP_Type | PyTorch版本 |
| ------- | -----  |   ---:  | ------ | -------: |  -------    |
| 1p-NPU  |   -    |  1574.1 |  1     |    O2    |   1.5    |
| 1p-NPU  |   -    |  2774.9 |  1     |    O2    |   1.8    |
| 8p-NPU  | 68.848 |  10912  |  30    |    O2    |   1.5    |
| 8p-NPU  | 69.813 | 10465.7 |  90    |    O2    |   1.8    |


# 版本说明

## 变更

2022.10.24：更新torch1.8版本，重新发布。

2021.07.13：首次发布  

## 已知问题

无。

