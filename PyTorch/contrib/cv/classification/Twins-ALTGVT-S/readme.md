# Twins-GVT-Small for PyTorch\_Owner

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)


# 概述

## 简述
Twins-GVT-S 对全局注意力策略进行了优化改进。全局注意力策略的计算量会随着图像的分辨率成二次方增长，因此如何在不显著损失性能的情况下降低计算量也是一个研究热点。Twins-SVT 提出新的融合了局部-全局注意力的机制，可以类比于卷积神经网络中的深度可分离卷积 （Depthwise Separable Convolution），并因此命名为空间可分离自注意力（Spatially Separable Self-Attention，SSSA）。与深度可分离卷积不同的是，Twins-SVT 提出的空间可分离自注意力是对特征的空间维度进行分组，并计算各组内的自注意力，再从全局对分组注意力结果进行融合。



- 参考实现：

```
url=https://github.com/Meituan-AutoML/Twins
branch=main
commit_id=4700293a2d0a91826ab357fc5b9bc1468ae0e987
model_name=PCPVT-Small
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
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/master/)       |

- 配置环境

  ```
  pip install -r requirements.txt
  ```



## 准备数据集

1.获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

该模型使用ImageNet2012，解压为如下格式：
```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
      ...
    class2/
      img2.jpeg
      ...
    ...
  val/
    class1/
      img3.jpeg
      ...
    class/2
      img4.jpeg
      ...
    ...
```
# 开始训练

## 训练模型
 
1. 进入解压后的源码包根目录。

  ```
   cd /${模型文件夹名称} 
  ```

2. 运行训练脚本。
  ```
   训练1p精度：
   bash ./test/train_finetune_1p.sh --data_path=xxx 
   训练1p性能：
   bash ./test/train_performance_1p.sh --data_path=xxx 
   训练8p精度：
   bash ./test/train_full_8p.sh --data_path=xxx 
   训练8p性能:
   bash ./test/train_performance_8p.sh --data_path=xxx 
  ```
3. 模型训练脚本参数说明如下。
  ```
  公共参数：
  --device                            //指定gpu或npu
  --data_path                         //数据集路径 
  --model                             //模型类型
  --batch-size                        //批大小
  --dist-eval                         //是否分布式评估
  --drop-path                         //dropout比率
  --epochs                            //批次
  --max_step                          //最大迭代次数
  ```

4. 日志和权重文件保存在如下路径。
  ```
  ./test/train_${device_id}.log          # training detail log
  ./test/Twins-GVT-Small_bs16_8p_acc.log             # 8p training performance result log
  ./output/ckpt                            # checkpoits
  ./test/Twins-GVT-Small_bs16_8p_acc.log        # 8p training accuracy result log
  ```
  # 训练结果展示

**表 2**  训练结果展示表

| 名称    |  FPS   |  Acc | 
| :------: | :------: | :------: |
| 1p-GPU  | 279 | / | 
| 1p-NPU  | 273 | / | 
| 8p-GPU | 2045  | 78.38 | 
| 8p-NPU  | 2138 | 78.43 | 







