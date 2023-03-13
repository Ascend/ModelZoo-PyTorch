# Twins-ALTGVT-S for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述
Twins-ALTGVT-S for PyTorch 对全局注意力策略进行了优化改进。全局注意力策略的计算量会随着图像的分辨率成二次方增长，因此如何在不显著损失性能的情况下降低计算量也是一个研究热点。Twins-SVT 提出新的融合了局部-全局注意力的机制，可以类比于卷积神经网络中的深度可分离卷积 （Depthwise Separable Convolution），并因此命名为空间可分离自注意力（Spatially Separable Self-Attention，SSSA）。与深度可分离卷积不同的是，Twins-SVT 提出的空间可分离自注意力是对特征的空间维度进行分组，并计算各组内的自注意力，再从全局对分组注意力结果进行融合。

- 参考实现：

```
url=https://github.com/Meituan-AutoML/Twins
commit_id=4700293a2d0a91826ab357fc5b9bc1468ae0e987
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
  | PyTorch 1.5 | torchvision==0.6.0；pillow==8.4.0 |
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


## 准备数据集

1. 获取数据集。

   下载开源数据集包括ImageNet2012，将数据集上传到服务器任意路径下并解压。

   数据集目录结构参考如下所示。

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

     启动单卡训练。

     ```
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能

     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --device                            //指定gp或npu
   --data-path                         //数据集径
   --model                             //模型类型
   --batch-size                        //批次大小
   --dist-eval                         //是否分式评估
   --momentum                          //动量
   --epochs                            //训练周期数
   --lr                                //初始学习率
   --weight-decay                      //权重衰减
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

| NAME | Acc@1 |  FPS   | Epochs | AMP_Type |  Torch_Version  |
| :------: | :------: | :------: | :------: | :------: | :------: |
| 1p-竞品V | -     | 279 | 1 | O1 | 1.5    |
| 8p-竞品V | 78.38  | 2045  | 100 | O1 | 1.5 |
| 1p-NPU  | -     | 296.83 | 2 | O1 | 1.8 |
| 8p-NPU  | 78.43 | 2026.39 | 100 | O1 | 1.8 |


# 版本说明

## 变更

2023.03.02：更新readme，重新发布。

2022.09.29：首次发布。

## FAQ

无。