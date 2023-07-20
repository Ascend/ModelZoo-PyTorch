# SE-ResNet-50 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

SE-ResNet是加入了“Squeeze-and-Excitation”（SE）模块的ResNet架构模型。SE模块能显式地建模特征通道之间的相互依赖关系。另外，SE-ResNet不引入一个新的空间维度来进行特征通道间的融合，而是采用了一种全新的“特征重标定”策略。具体来说，就是通过学习的方式来自动获取到每个特征通道的重要程度，然后依照这个重要程度去提升有用的特征并抑制对当前任务用处不大的特征。SE模块以最小的额外计算成本为深层架构带来了显著的性能改进。

- 参考实现：

  ```
  url=https://github.com/hujie-frank/SENet
  commit_id=0262d43d44c561fd53c3dba210cc8bacfc60500d
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
  | PyTorch 1.5 | torchvision==0.6.0 |
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

   用户自行获取原始数据集，使用的开源数据集为ImageNet，将数据集上传到服务器任意路径下并解压。
   
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
   > 该数据集的训练过程脚本只作为一种参考示例

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
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh --data_path=/data/xxx/  # 启动评测脚本前，需对应修改评测脚本中的--resume参数，指定ckpt文件路径
     ```

   - 多机多卡性能数据获取流程
     ```
     1. 安装环境
     2. 开始训练，每个机器请按下面提示进行配置
      bash ./test/train_performance_multinodes.sh --data_path=数据集路径 --batch_size=单卡batch_size*所有卡数 --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。
   
   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --addr                              //主机地址
   --workers                           //加载数据进程数
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小
   --learning-rate                     //初始学习率
   --mom                               //动量
   --weight-decay                      //权重衰减
   --amp                               //是否混合精度，默认：False
   --device                            //使用设备，gpu或npu
   多卡训练参数：
   --multiprocessing-distributed       //是否使用多卡训练
   ```

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V | - | - | 1 | - | 1.5 |
| 8p-竞品V | - | - | 100 | - | 1.5 |
| 1p-NPU |   -    | 747.41  |   1    |    O2    |  1.8  |
| 8p-NPU | 77.521 | 5306.66 |  100   |    O2    |  1.8  |

# 版本说明

2022.08.01：更新pytorch1.8版本，重新发布。

## FAQ
无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md