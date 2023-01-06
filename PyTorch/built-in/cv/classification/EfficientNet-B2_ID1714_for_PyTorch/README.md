# EfficientNet-B2 for PyTorch

-   [概述](#1)
-   [准备训练环境](#2)
-   [开始训练](#3)
-   [训练结果展示](#4)
-   [版本说明](#5)



# 概述<a id="1"></a>

## 简述

EfficientNet是一个新的卷积网络家族，与之前的模型相比，具有更快的训练速度和更好的参数效率。
该模型通过一组固定的缩放系数统一缩放这在网络深度，网络宽度，分辨率这三方面有明显优势。
在EfficientNet中，这些特性是按更有原则的方式扩展的，也就是说，一切都是逐渐增加的。

- 参考实现：

  ```
  url=https://github.com/lukemelas/EfficientNet-PyTorch
  commit_id=7e8b0d312162f335785fb5dcfa1df29a75a1783a
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

# 准备训练环境<a id="2"></a>

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                                           |
  |------------------------------------------------------------------------------| ------------------------------------------------------------ |
  | 硬件 | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)  |
  | NPU固件与驱动 | [6.0.rc1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)                       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集imagenet2012，将数据集上传到服务器并解压。

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


# 开始训练<a id="3"></a>

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
     bash ./test/train_full_1p.sh --data_path=real_data_path  # 1p精度    
     bash ./test/train_performance_1p.sh --data_path=real_data_path  # 1p性能
     ```
   
   - 单机8卡训练
   
     启动8卡训练。
   
     ```
     bash ./test/train_full_8p.sh --data_path=real_data_path  # 8p精度
     bash ./test/train_performance_8p.sh --data_path=real_data_path  # 8p性能 
     ```

   其中real_data_path参数填写数据集路径。
   
   
   - 多机多卡性能数据获取流程

	```
		1.安装环境
		2.开始训练，每个机器所请按下面提示进行配置
			bash ./test/train_performance_multinodes.sh --data_path=数据集路径 --batch_size=单卡batch_size * 单机卡数 --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
	```

   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --data                              //数据集路径
   --arch                              //使用模型，默认：efficientnet-b2
   --epoch                             //重复训练次数
   --batch-size                        //训练批次大小
   --lr                                //初始学习率，默认：0.1
   --momentum                          //动量，默认：0.9
   --weight_decay                      //权重衰减，默认：0.0001
   --amp                               //是否使用混合精度
   --loss_scale                        //混合精度lossscale大小
   --pm                                //混合精度类型
   ```
   

# 训练结果展示<a id="4"></a>

**表 2**  训练结果展示表

| NAME   | Acc@1  | FPS   | Epochs |  Torch_version |
|--------|--------|:------|--------| :------------ |
| 1p-竞品  | -      | -     | -      |  -             |
| 8p-竞品  | -      | -     | -      |  -             |
| 1p-NPU | -      | 701.2 | 1      |  1.8           |
| 8p-NPU | -      | 4505  | 1      |  1.8           |


# 版本说明<a id="5"></a>

## 变更

2022.12.30：更新Readme。

## 已知问题

无。

