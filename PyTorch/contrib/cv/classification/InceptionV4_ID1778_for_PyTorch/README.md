# InceptionV4_ID1778_for_PyTorch
-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

GoogLeNet对网络中的传统卷积层进行了修改，提出了被称为Inception的结构，用于增加网络深度和宽度，提高深度神经网络性能。从InceptionV1到InceptionV4有4个更新版本，每一版的网络在原来的基础上进行改进，提高网络性能。InceptionV4研究了Inception Module和Reduction Module的组合，通过多次卷积和非线性变化，极大的提升了网络性能。

- 参考实现：

  ```
  url=https://github.com/Cadene/pretrained-models.pytorch#inception
  commit_id=8aae3d8f1135b6b13fed79c1d431e3449fdbf6e0
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

  | 配套        | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动  | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)|

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，CIFAR-10等，将数据集上传到服务器任意路径下并解压。

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
   >数据集路径以用户自行定义的路径为准

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
   --workers                           //加载数据进程数      
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小
   --lr                                //初始学习率，默认：0.1
   --momentum                          //动量，默认：0.9
   --weight_decay                      //权重衰减，默认：0.0001
   --amp                               //是否使用混合精度
   --loss-scale                        //混合精度lossscale大小
   --opt-level                         //混合精度类型
   --start-epoch                       //开始轮次
   --pretrained                        //预训练
   --multiprocessing-distributed       //是否使用多卡训练
   --device-list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1   |  FPS     | Epochs | AMP_Type | Torch_version |
| ------- | ------- | -------: | ------ | -------: | ------------- |
| 1p-竞品 | -       | 210.508  | 1      |        - | -             |
| 8p-竞品 | 79.7586 | 1617.189 | 240    |        - | -             |
| 1p-NPU  | -       | 320.095  | 1      |       O2 | 1.5           |
| 1p-NPU  | -       | 474.607  | 1      |       O2 | 1.8           |
| 8p-NPU  | 79.259  | 1976.6   | 240    |       O2 | 1.5           |
| 8p-NPU  | 79.574  | 2366.193 | 240    |       O2 | 1.8           |


# 版本说明

## 变更

2022.09.09：更新pytorch1.8版本，重新发布。

2021.07.14：首次发布。

## 已知问题

无。