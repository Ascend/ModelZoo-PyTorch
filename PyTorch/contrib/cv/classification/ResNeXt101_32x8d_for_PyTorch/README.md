# ResNeXt101_32x8d for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述
传统的要提高模型的准确率，都是加深或者加宽网络，但是随着超参数数量的增加（比如channels数，filter size等等），网络设计的难度和计算的开销也会增加。而ResNeXt结构可以在不增加参数复杂度的前提下提高准确率，同时还减少了超参数的数量。ResNeXt同时采用VGG堆叠思想和Inception的split-transform-merge思想，可扩展性比较强，可以认为是在增加准群率的同时基本不改变或者降低模型的复杂度。

- 参考实现：

  ```
  url=https://github.com/pytorch/examples/tree/master/imagenet
  commit_id=49e1a8847c8c4d8d3c576479cb2fe2fd2ac583de
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
  | 硬件       | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，将数据集上传到服务器任意路径下并解压。

   以ImageNet2012数据集为例，数据集目录结构参考如下所示。

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
   --addr                              //主机地址
   --seed                              //初始化训练种子
   --workers                           //加载数据进程数    
   --learning-rate                     //初始学习率 
   --print-freq                        //打印频率
   --dist-backend='hccl'               //通信后端
   --batch-size                        //训练批次大小
   --epoch                             //重复训练次数
   --warm_up_epochs                    //warm up
   --rank                              //节点编号
   --amp                               //是否使用混合精度
   --momentum                          //动量
   --wd                                //权重衰减
   --gpu                               //卡号
   ```

　
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1  | FPS     | Epochs | AMP_Type | Torch_version |
| ------- | ------ | :------ | ------ | :------- | :------------ |
| 1p-竞品 | -      | 126     | 1      | -        | -             |
| 8p-竞品 | 79.22  | 1200    | 90     | -        | -             |
| 1p-NPU  | -      | 482     | 1      | O2       | 1.8           |
| 8p-NPU  | 79.422 | 3765    | 90     | O2       | 1.8           |



# 版本说明

## 变更

2020.12.10：首次发布

## 已知问题

无。











