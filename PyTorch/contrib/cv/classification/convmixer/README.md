# Convmixer for PyTorch 

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述 

## 简述

Convmixer由一个patch embedding层和一个简单的全卷积块的重复应用组成，并保持着patch embedding的空间结构。Convmixer是一个非常简单的卷积架构，直接在patch上操作，它在所有层中保持相同分辨率和大小的表示。Convmixer证明了patch表示本身可能会是Vision transformer这样的新架构卓越性能的最关键组件。

- 参考实现：
  ```
  url=https://github.com/locuslab/convmixer.git
  commit_id=47048118e95721a00385bfe3122519f4b583b26e
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
  ```

- 通过Git获取代码方法如下：

  ```
  git clone {url} # 克隆仓库的代码
  cd {code_path}  # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境
- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1** 版本配套表
  | 配套       | 版本      |
  | :---------- | :------------------------------------------------------------ |
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)|


- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```
  pip install -r requirements.txt
  ```
## 准备数据集
1. 获取数据集 

     下载[ImageNet](https://image-net.org/)数据集，将数据集上传到服务器任意路径下并解压。
     数据集目录结构参考如下所示。
     ```
    ├── imagenet
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

   该模型支持单机单卡性能和单机8卡训练。
    - 单机单卡性能
    
      启动单卡性能测试。

      ```
      bash ./test/train_performance_1p.sh --data_path=xxx   
      ```
    - 单机8卡性能
    
      启动8卡性能测试。

      ```
      bash ./test/train_performance_8p.sh --data_path=xxx   
      ```

    - 单机8卡训练

      启动8卡训练。
      ```
      bash ./test/train_full_8p.sh --data_path=xxx
      ```

    --data\_path参数填写数据集路径。

    模型训练脚本参数说明如下。
    ```
    公共参数：
    --model                             //网络模型
    --opt                               //优化器
    --input-size                        //输入图片大小
    --cutmix                            //数据增强
    --mixup                             //数据增强  
    --epochs                            //重复训练次数
    --b                                 //训练批次大小
    --lr                                //初始学习率，默认：0.01
    --num-classes                       //分类数
    --amp                               //是否使用混合精度
    --device                            //指定训练用卡
    ```
  
    训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|  name   | Acc@1   |   FPS   |   Epochs   | AMP_Type | Pytorch |
|:-------:| :-----: |:-------:|:----------:| :------: | :-----: |
| 1p-NPU  |   -     |  50.24  |     1      |    O2    |   1.5   |
| 1p-NPU  |   -     |  42.09  |     1      |    O2    |   1.8   |
| 8p-NPU  | 80.2%   | 407.18  |    150     |    O2    |   1.5   |
| 8p-NPU  | 80.2%   | 376.64  |    150     |    O2    |   1.8   |

# 版本说明

## 变更

2022.11.02：更新内容，重新发布。

2020.07.08：首次发布。

### 已知问题

无。

