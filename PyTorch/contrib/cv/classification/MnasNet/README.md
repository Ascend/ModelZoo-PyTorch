# MnasNet for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

MnasNet是Google研究小组2019年在论文《MnasNet: Platform-Aware Neural Architecture Search for Mobile》中推出的新模型。Mnasnet网络是介于mobilenetV2和mobilenetV3之间的一个网络，这个网络是采用强化学习搜索出来的一个网络，是谷歌提出的一个轻量化网络。

- 参考实现：

  ```
  url=https://github.com/pytorch/vision/blob/master/torchvision/models/mnasnet.py
  commit_id=91e03b91fd9bab19b4c295692455a1883831a932
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

   用户自行获取原始数据集imagenet，将数据集上传到服务器任意路径下并解压。

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
      
      bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
      ```

   - 单机8卡训练
   
     启动8卡训练。
   
     ```
      bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
      
      bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8p性能
     ```

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh --data_path=/data/xxx/   
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。
   
   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --device                            //使用设备，gpu或npu
   --workers                           //加载数据进程数      
   --epochs                             //重复训练次数
   --batch-size                        //训练批次大小
   --lr                                //初始学习率，默认：0.1
   --momentum                          //动量，默认：0.9
   --weight-decay                      //权重衰减，默认：0.0001
   多卡训练参数：
   --multiprocessing-distributed       //是否使用多卡训练
   --device-list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME | Acc@1  | FPS  | Epochs | AMP_Type | Torch_Version |
| :----: | :--: | :----: | :------: | :---: | :---: |
| 1p-竞品V | - | 809 | 6 | O2 | 1.5 |
| 8p-竞品V | 73.046 | 1408 | 300 | O2 | 1.5 |
| 1p-NPU |   -    | 2569.8 |   6   |    O1    |  1.8  |
| 8p-NPU | 72.819 | 14413.78 |  300   |    O1    |  1.8  |

# 版本说明

## 变更

2023.03.02：更新readme，重新发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md