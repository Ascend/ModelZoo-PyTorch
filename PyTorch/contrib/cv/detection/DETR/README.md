# DETR for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述

## 简述

DETR提出了一种将对象检测视为直接集合预测问题，能够一次性预测所有的目标，其训练采用一种集合损失函数以端到端方式进行，集合损失定义在预测结果与真实目标的二部图匹配结果上；该方法简化了检测管道，有效地消除了对许多手工设计组件的需求，例如非最大抑制程序或锚点生成，简化了检测流程；和存在的其他检测方法不一样，DETR不需要任何定制的层，因此能够便捷的在任何包含transformer和CNN的深度框架中进行复现。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/detr
  commit_id=585ce2c4fb80ae6ab236f79f06911e2f8bef180c
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
  | PyTorch 1.5 | torchvision==0.2.2.post3; pillow==8.4.0|
  | PyTorch 1.8 | torchvision==0.9.1; pillow==9.1.0  |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # Pytorch1.5版本

  pip install -r 1.8_requirements.txt  # Pytorch1.8版本
  ```
  > **说明:**
  > 只需执行一条对应的PyTorch版本依赖安装命令。


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集COCO2017，将数据集上传到服务器任意路径下并解压。

   目录结构如下：
    ```bash
      coco
        ├── annotations
        ├── train2017
        ├── val2017
    ```
   > **说明：**
   >该数据集的训练过程脚本只作为一种参考示例。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   dos2unix ./test/*.sh
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练
     ```
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度

     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --coco_path                         //数据集路径
   --workers                           //加载数据进程数
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小
   --gpu                               //指定设备号
   --max_steps                         //跑性能最大执行步数
   ```

 训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

训练细节：

我们使用 AdamW 训练 DETR，将`transformer`中的学习率设置为 1e-4 和主干中的 1e-5。
水平翻转、缩放和裁剪用于数据增强。
图像将重新缩放为最小大小 800，最大大小为 1333。
`transformer`的`dropout`比例为 0.1，整个模型都是用梯度裁剪为 0.1 进行训练。

# 训练结果展示

**表 2**  训练结果展示表


| Name    |  LOSS   |   FPS    |  Epochs/steps   |  AMP_Type | Torch_version |
| ------- |-------: |  ------- |   -----         | ------    | ------------  |
|1p-竞品v | -       |   12.7   |  1000steps      |  O0       |    1.5       |
|8p-竞品v | -       |   73.5   |   2epochs       |  O0       |    1.5       |
|1p-NPU   | 24.4104 |   0.2    |  1000steps      |  O0       |    1.5       |
|8p-NPU   | 24.9557 |   0.489  |  2 epochs       |  O0       |    1.5       |

# 版本说明

## 变更

2020.07.08：首次发布。

## 已知问题

无。


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
