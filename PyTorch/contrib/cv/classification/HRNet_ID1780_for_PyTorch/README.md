# HRNet for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述
HRNet，是一个用于图像分类的高分辨网络。通过并行连接高分辨率到低分辨率卷积来保持高分辨率表示，并通过重复跨并行卷积执行多尺度融合来增强高分辨率表示。在像素级分类、区域级分类和图像级分类中，证明了这些方法的有效性。其创新点在于能够从头到尾保持高分辨率，而不同分支的信息交互是为了补充通道数减少带来的信息损耗，这种网络架构设计对于位置敏感的任务会有奇效。

- 参考实现：

  ```
  url=https://github.com/HRNet/HRNet-Image-Classification
  commit_id=f760c988482cdb8a1f69b10b219d669721144582
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```
  pip install -r requirements.txt
  ```

## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，将数据集上传到服务器任意路径下并解压。 数据集目录结构如下所示：

   ```
   ├── ImageNet2012
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

     启动单卡训练
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
   - 单机单卡评测

     启动单卡评测。
     ```
     bash ./test/train_eval_1p.sh --data_path=xxx --device_id=xxx  # 单卡评测
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   --device_id参数填写使用哪个设备。
   
   脚本中resume默认关闭，若要开启请在yaml配置文件中修改RESUME的值为true.

   模型训练脚本参数说明如下。

   ```
   公共参数：
   data_path                           //数据集路径
   --cfg                               // 模型配置文件
   --addr                              // master节点 地址
   --nproc                             // 启动节点数量
   --lr                                // 学习率
   --workers                           // 加载数据进程数
   --train_epochs                      // 重复训练次数
   --bs                                // 训练批次大小
   --device_id                         // 指定设备号
   --stop_step                         // 跑性能最大执行步数
   --dn                                // 当前节点排名
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V |   -   | -    |   1    |    -     |      1.5      |
| 8p-竞品V |   -   |  -   |  100   |    -     |      1.5      |
|  1p-NPU  |   -   | 84.1  |   1    |    O2    |      1.8      |
|  8p-NPU  |  76.65  | 533.2  |  100   |    O2    |      1.8      |


# 版本说明

## 变更

2022.07.08：首次发布。

## 已知问题

无。









