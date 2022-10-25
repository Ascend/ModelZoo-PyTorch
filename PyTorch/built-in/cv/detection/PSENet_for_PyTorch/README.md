# PSENet for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

PSENet是一个用于自然文本检测的网络模型，可对任意形状的文本进行有效检测。该网络预测每个文本行的不同尺度的kernels，并对这些kernels采用基于BFS的渐进式尺度扩张算法，可有效解决任意形状文本检测问题和相邻文本混淆问题。

- 参考实现：

  ```
  url=https://github.com/whai362/psenet.git
  commit_id=e6686b143a8c0520971b4ef8587250c72e2f05f4
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/detection
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}          # 克隆仓库的代码
  cd {code_path}     	  # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括[ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=downloads)、[ICDAR2017](https://rrc.cvc.uab.es/?ch=4&com=downloads)等，将数据集上传到服务器任意路径下并解压。

   以ICDAR2015数据集为例，数据集目录结构参考如下所示。

   ```
   ├── ICDAR2015
         ├──train
              ├──图片
                    │──图片1
                    │──图片2
                    │   ...       
              ├──标签
                    │──标签1
                    │──标签2
                    │   ...
              ├──...
         ├──val  
              ├──图片
                    │──图片1
                    │──图片2
                    │   ...
              ├──标签
                    │──标签1
                    │──标签2
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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/    
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/   
     ```

   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data                              //数据集路径
   --addr                              //主机地址
   --arch                              //使用模型，默认：PSENet
   --workers                           //加载数据进程数
   --epoch                             //重复训练次数
   --batch-size                        //训练批次大小
   --lr                                //初始学习率，默认：0.001
   --momentum                          //动量，默认：0.9
   --weight_decay                      //权重衰减，默认：0.0001
   --amp                               //是否使用混合精度
   --loss-scale                        //混合精度lossscale大小
   --opt-level                         //混合精度类型
   多卡训练参数：
   --multiprocessing-distributed       //是否使用多卡训练
   --device-list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1   |  FPS   | Epochs | AMP_Type | Torch_version |
| ------- | -----   | ---:   | ------ | -------: | -----------:  |
| 1p-NPU  | -       |  9.123 | 1      |        - | Torch1.5      |
| 1p-NPU  | -       | 14.945 | 1      |       O2 | Torch1.8      |
| 8p-NPU  | 91.770  | 11.791 | 600    |        - | Torch1.5      |
| 8p-NPU  | 94.040  | 56.635 | 600    |       O2 | Torch1.8      |




# 版本说明

## 变更

2022.10.18：更新内容，重新发布。

2022.06.08：首次发布。

## 已知问题


无。











