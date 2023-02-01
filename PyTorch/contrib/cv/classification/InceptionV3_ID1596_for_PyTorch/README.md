# Inception_v3 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

GoogLeNet对网络中的传统卷积层进行了修改，提出了被称为Inception的结构，用于增加网络深度和宽度，提高深度神经网络性能。从Inception V1到Inception V4有4个更新版本，每一版的网络在原来的基础上进行改进，提高网络性能。Inception V3研究了Inception Module和Reduction Module的组合，通过多次卷积和非线性变化，极大的提升了网络性能。

- 参考实现：

  ```
  url=https://github.com/pytorch/examples.git
  commit_id=507493d7b5fab51d55af88c5df9eadceb144fb67
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
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
  注意:pillow建议安装更新的版本。如果无法直接安装对应版本的torchvision，可以使用源代码安装对应版本。源代码参考链接:https://github.com/pytorch/vision， 建议pilow为9.1.0，torchvision为0.6.0

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
   - 在线推理例子
      ```
      python3.7.5 demo.py
      ```
   - 导出ONNX
      ```
      python3.7.5 pthtar2onnx.py
      ```
   - 多机多卡性能数据获取流程
     ```
     1. 多机环境准备
     2. 开始训练，每个机器所请按下面提示进行配置
            bash ./test/train_performance_multinodes.sh --data_path=数据集路径 --batch_size=单卡batch_size*单机卡数 --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
     ```
   --data_path：数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   -a                             // 模型名称
   --data                         // 数据集路径
   -j                             // 最大线程数
   --output_dir                   // 输出目录
   -b                             // 训练批次大小
   --lr                           // 初始学习率
   --print-freq                   // 打印频率
   --epochs                       // 重复训练次数
   --label-smoothing              // 标签平滑系数
   --wd                           // 权重衰减系数
   -p                             // 类别数量
   --amp                          // 使用混合精度
   --npu                          // 使用设备
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

## Inception_v3 training result

| NAME | Acc@1  | FPS | Epochs   | AMP_Type |
| :------: | :------: | :------: | :------: | :------: |
| 1p-1.5   | - | 295  | 240     | O2       |
| 1p-1.8   | - | 942.288 | 240      | O2       |
| 8p-1.5   | 78.49 | 3251  | 240     | O2       |
| 8p-1.8   | 78.12 | 6822.728 | 240      | O2       |
# 版本说明

## 变更

2022.09.24：首次发布。

## 已知问题

无。

