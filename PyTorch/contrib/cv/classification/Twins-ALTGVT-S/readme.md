# Twins-ALTGVT-S for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)


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



- 通过Git获取代码方法如下：

```
git clone {url}        # 克隆仓库的代码
cd {code_path}         # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
```

- 通过单击“立即下载”，下载源码包

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 硬件       | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)   |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```
  pip install -r requirements.txt
  ```

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
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/

     bash ./test/train_full_8p.sh --data_path=/data/xxx/

     ```
   - 微调脚本
     ```
      bash ./test/train_finetune_1p.sh --data_path=/data/xxx/ --finetune_pth=预训练模型路径
     ```

  --data_path：数据集路径

  --fine_tune_path：预训练的模型路径

  模型训练脚本参数说明如下。

    --device                            //指定gp或npu
    --data_path                         //数据集径
    --model                             //模型类型
    --batch-size                        //批大小
    --dist-eval                         //是否分式评估
    --drop-path                         //dropou比率
    --epochs                            //批次
    --max_step                          //最大迭代次数

  日志和权重文件保存在如下路径。

    ./test/train_${device_id}.log          # training detail log

    ./test/Twins-GVT-Small_bs16_8p_acc.log             # 8p training performance result log

    ./output/ckpt                            # checkpoits

    ./test/Twins-GVT-Small_bs16_8p_acc.log        # 8p training accuracy result log
  # 训练结果展示

**表 2**  训练结果展示表

| 名称    |  FPS   |  Acc |  Npu Torch版本  |
| :------: | :------: | :------: | :------: |
| 1p-GPU  | 279 | / | /   |
| 1p-NPU  | 296.83 | / | 1.8 |
| 8p-GPU | 2045  | 78.38 | / |
| 8p-NPU  | 2026.39 | 78.43 | 1.8 |

# 版本说明

## 变更

2022.09.29：首次发布。
## 已知问题

无。






