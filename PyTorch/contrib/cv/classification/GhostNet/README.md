# GhostNet for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

  因为有限的内存和计算资源，在嵌入式设备中部署卷积神经网络（CNNs）是困难的。特征图的冗余是那些成功CNNs的一个重要特征，但是很少有关于网络架构设计的研究。Ghost模型可以从廉价的运算中得到更多的特征图。基于特征图集，我们应用一系列低成本的线性运算去生成许多重影特征图，它能够完全的揭露隐藏在本特征图下的信息。这个提出来的Ghost模块可以当成是一个即插即拨的组件，去升级已有的卷积神经网络。
- 参考实现：

  ```
  url=https://github.com/huawei-noah/CV-Backbones/tree/master 
  commit_id=3e7700491d582c1c35f6ae55ae08e4658823a2f7
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
  | PyTorch 1.5 | torchvision==0.2.2.post3 |
  | PyTorch 1.8 | torchvision==0.9.1 |

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

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，将数据集上传到服务器任意路径下并解压。

   以ImageNet数据集为例，数据集目录结构参考如下所示。

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

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/ # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/ # 8卡性能
     ```

   - 单机单卡评测

     启动单卡评测。

     ```
     bash ./test/train_eval_1p.sh --data_path=/data/xxx/ --checkpoint=real_pre_train_model_path # 单卡评测
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。
   
   --checkpoint参数填写训练权重生成路径，需写到权重文件的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --amp                               //是否使用混合精度
   ${data_path}                        //数据集路径
   --model                             //使用模型
   --momentum                          //动量   
   --workers                           //加载数据进程数
   --lr                                //初始学习率
   --weight-decay                      //权重衰减
   --sched                             //学习率调整策略
   --j                                 //加载数据进程数
   --epochs                            //重复训练次数
   -b                                  //训练批次大小
   --warmup-lr                         //训练初期的学习率
   --dorp                              //神经元删除率
   --npu                               //使用的npu设备号
   --bum-classes                       //分类数
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V |   -   | -  |   10   |    -     |      1.5      |
| 8p-竞品V | - | - |  400   |    -     |      1.5      |
|  1p-NPU  |   -   | 1378.8  |   10   |    O2    |      1.8      |
|  8p-NPU  | 73.129 | 9559.2  |  400   |    O2    |      1.8      |


# 版本说明

## 变更

2023.1.30：更新readme，重新发布。


## FAQ

无。


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
