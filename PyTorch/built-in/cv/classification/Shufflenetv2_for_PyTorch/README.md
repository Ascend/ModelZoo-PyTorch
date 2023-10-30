# ShuffleNetV2 for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述
shuffleNetV2是一个改进shuffleNetV1的轻量级的网络，为了解决在有限计算资源下特征通道数量不够多的问题，引入了一个简单的通道分离的操作，使得shuffleNetV2在很小的计算成本下性能优于其它网络。

- 参考实现：

  ```
  url=https://github.com/megvii-model/ShuffleNet-Series.git
  commit_id=d69403d4b5fb3043c7c0da3c2a15df8c5e520d89
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
  | PyTorch 1.11 | pillow==9.1.0 |
  | PyTorch 2.1 | pillow==9.1.0 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本

  pip install -r 1.11_requirements.txt  # PyTorch1.11版本

  pip install -r 2.1_requirements.txt  # PyTorch2.1版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。


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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data                              //数据集路径
   --addr                              //主机地址
   --seed                              //初始化训练种子
   --workers                           //加载数据进程数    
   --learning-rate                     //初始学习率 
   --print-freq                        //打印频率
   --eval-freq                         //测试周期
   --arch                              //所选模型架构
   --dist-backend='hccl'               //通信后端
   --batch-size                        //训练批次大小
   --epoch                             //重复训练次数
   --warm_up_epochs                    //warm up
   --rank                              //节点编号
   --amp                               //是否使用混合精度
   --momentum                          //动量
   --wd                                //权重衰减
   --device-list                       //卡号
   --benchmark                         //设置benchmark状态
   --device_num                        //使用卡数
   --dist-url                          //设置分布式训练的网址
   --multiprocessing-distributed       //使是否使用多卡训练
   --world-size                        //分布式训练节点数量
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1  | FPS     | Epochs | AMP_Type | Torch_version |
| :-----: | :----: | :-----: | :----: | :------: | :-----------: |
| 1p-竞品V | -      | -       | -      | -        | -             |
| 8p-竞品V | -      | -       | -      | -        | -             |
| 1p-NPU  | -      | 5038.508 | 1      | O2       | 1.8           |
| 8p-NPU  | 65.158 | 17420.493 | 240    | O2       | 1.8           |

# 版本说明

## 变更

2023.02.15：更新readme，重新发布。

2020.07.08：首次发布。

## FAQ

1.某些版本的numpy会引发错误，请避免使用该版本：1.19.2。

2.截至目前，Ascend Pytorch使用连续操作仍然效率低下，因此ShufflenetV2使用自定义方法实现，更多细节参阅models/shufflenetv2_wock_op_woct.py。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
