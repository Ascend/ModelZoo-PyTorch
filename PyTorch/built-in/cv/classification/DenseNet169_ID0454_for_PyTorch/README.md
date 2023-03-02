# DenseNet169 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

DenseNet-169是一个经典的图像分类网络，对于一个L层的网络，DenseNet共包含L*（L+1）/2个连接，相比ResNet，这是一种密集连接，他的名称也由此而来，另一大特色为通过特征在channel上的连接来实现特征重用（feature reuse），这些特点让DenseNet在参数和计算成本更少的情形下实现比ResNet更优的性能。

- 参考实现：

  ```
  url=https://github.com/pytorch/vision.git
  commit_id=882e11db8138236ce375ea0dc8a53fd91f715a90
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | pillow==8.4.0 |
  | PyTorch 1.8 | pillow==9.1.0 |
  
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

   用户自行获取原始数据集，下载ImageNet2012数据集，将数据集上传到服务器任意路径下并解压。

   数据集目录结构参考如下所示：

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

     ```shell
     bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```shell
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data-path                         //数据集路径
   --model                             //使用模型，默认：densenet169
   --device_id                         //指定训练所使用的npu device卡id
   --batch-size                        //训练批次大小
   --epochs                            //重复训练次数
   --workers                           //加载数据进程数
   --lr                                //初始学习率，默认：0.1
   --momentum                          //动量，默认：0.9
   --weight-decay                      //权重衰减，默认：0.0004
   --distributed                       //是否使用多卡训练
   --apex                              //是否使用混合精度
   --apex-opt-level                    //混合精度类型
   --loss_scale_value                  //混合精度lossscale大小
   --seed                              //设置随机数
   --print-freq                        //输出模型训练精度和性能信息
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 3**  训练结果展示表

| NAME    | Acc@1  | FPS     | Epochs | AMP_Type | Torch_version |
| :-----: | :----: | :-----: | :----: | :------: | :-----------: |
| 1p-NPU  | - | 637.079 |   1    | O2       | 1.8           |
| 8p-NPU  | 73.726 | 5143.396 | 90     | O2       | 1.8           |


# 版本说明

## 变更

2023.02.21：更新readme，重新发布。

2022.03.18：首次发布。

## FAQ

无。
