# EfficientNetV2 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

EfficientNetV2是Efficient的改进版，accuracy达到了发布时的SOTA水平，而且训练速度更快参数来更少。相对EfficientNetV1系列只关注准确率，参数量以及FLOPs，V2版本更加关注模型的实际训练速度。


- 参考实现：

  ```
  url=https://github.com/rwightman/pytorch-image-models/blob/v0.5.4/timm/models/efficientnet.py
  commit_id=9ca343717826578b0e003e78b694361621c2b0ef
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
  | 固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |
  | Apex       | [0.1](https://gitee.com/ascend/apex/tree/master/)            |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，CIFAR-10等，将数据集上传到服务器任意路径下并解压。

   EfficientNetV2迁移使用到的ImageNet2012数据集目录结构参考如下所示。

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
   > 该数据集的训练过程脚本只作为一种参考示例。




# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡完整精度训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path="/data/xxx/" 
     ```

   - 单机8卡完整精度训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path="/data/xxx/"
     ```
   - 单机单卡性能测试

     启动单卡训练。

     ```
     bash ./test/train_performance_1p.sh --data_path="/data/xxx/" 
     ```

   - 单机8卡性能测试

     启动8卡训练。

     ```
     bash ./test/train_performance_8p.sh --data_path="/data/xxx/"
     ```
   

--data_path参数填写数据集路径。

模型训练脚本参数说明如下。

   ```
   --model                           # 模型结构
   -b                                # batch size
   --sched step                      # 学习率调涨策略 
   --epochs                          # 总epoch数
   --decay-epochs 2.4                # 学习率下降epoch间隔
   --decay-rate .97                  # 学习率下降比例
   --opt rmsproptf                   # 优化器
   --opt-eps .001                    # 优化器阈值
   -j 8                              # worker数
   --warmup-lr 4e-6                  # warmup初始学习率
   --weight-decay 1e-5               # 权重正则化参数
   --drop 0.3 \                      # dropout的概率
   --drop-connect 0.2 \              # dropo connect的概率
   --aa rand-m9-mstd0.5 \            # auto augment策略
   --lr .112 \                       # 基础学习率
   --log-interval 1 \                # 打印间隔
   --img-size 288 \                  # 输入图像大小
   --apex-amp \                      # 混合精度方法
   --model-ema \                     # 是否开启ema
   --model-ema-decay 0.9999 \        # ema滑动平均系数
   --mixup 0.2 >                     # mixup系数
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表
| NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch |
| ------ | ----- | ---- | ------ | -------- | ----- |
| 1p-GPU | -     | 533  | -      | O1       | 1.8   |
| 1p-NPU | -     | 602  | -      | O1       | 1.8   |
| 8p-GPU | 82.34 | 4100 | 350    | O1       | 1.8   |
| 8p-NPU | 82.19 | 4500 | 350    | O1       | 1.8   |


# 版本说明

## 变更

2022.10.14：首次发布。

## 已知问题

无。