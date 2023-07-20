# RegNetX for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述
## 简述

RegNetX-1.6GF是FAIR提出的一种RegNetX。RegNetX使用了新的网络设计范式，结合了手动设计网络和神经网络搜索（NAS）的优点。和手动设计网络一样，其目标是可解释性，可描述一些简单网络的一般设计原则，并在各种设置中泛化；又和NAS一样能够利用半自动过程来找到易于理解、构建和泛化的简单模型。
- 参考实现：

  ```
  url=https://github.com/facebookresearch/pycls
  commit_id=2647af9f32eb27d099bd852fb11f731876316757
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
  | PyTorch 1.5 | torchvision==0.2.2.post3；pillow==8.4.0 |
  | PyTorch 1.8 | torchvision==0.9.1；pillow==9.1.0 |

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

   用户自行下载 `ImageNet` 数据集，将数据集上传到服务器任意路径下并解压。
    
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
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/   # 8卡精度
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/ # 8卡性能
     ```

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh --data_path=/data/xxx/ --pth_path=real_pre_train_model_path # 8卡评测
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。
   
   --pth_path参数填写训练权重生成路径，需写到权重文件的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data                         //数据集路径
   --train-batch                  //训练批次大小
   --lr                           //初始学习率
   --wd                           // 权重衰减参数
   -c                             // 检查点路径
   --print-freq                   //打印频率
   --epochs                       //重复训练次数
   --opt-level                    // apex优化器级别
   --loss-scale                   // apex loss_scale值
   --device-id                    //使用设备
   --j                            //加载数据线程数
   --label-smoothing              // 标签平滑系数
   --warmup                       // warmup系数
   --rank                         // 进程全局排名
   --local_rank                   // 进程局部排名
   --world_size                   // 总进程数
   --log-name                     // 日志文件名字
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1  |    FPS  | Epochs | AMP_Type | Torch_Version |
| :-----: | :----: | :----:  | :----: | :------: | :--------:  |
| 1p-竞品V| - | 640 | 1 | - | 1.5 |
| 8p-竞品V| 77.17 | 4600 | 100 | - | 1.5 |
| 1p-NPU-ARM  |   -    |   1824.49   | 1      |    O2    |   1.8    |
| 8p-NPU-ARM  | 77.34    |  14982.96    |  100   |  O2   |    1.8   |
| 1p-NPU-非ARM  |   -    |   1836.394  | 1      |    O2    |   1.8    |
| 8p-NPU-非ARM  | -    |  8605.472    |  100   |  O2   |    1.8   |

# 版本说明

## 变更

2022.10.24：更新torch1.8版本，重新发布。

2020.12.19：首次发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md