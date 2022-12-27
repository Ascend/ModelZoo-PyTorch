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
  
- 通过Git获取代码方法如下：

  ```
  git clone {url} # 克隆仓库的代码
  cd {code_path} # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
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

    注意：pillow 建议安装较新的版本。如果无法直接安装对应的torchvision版本，可以使用源码安装对应的版本。源码参考链接：[https ://github.com/pytorch/vision](https://github.com/pytorch/vision)。
  
    建议：pillow==9.1.0，torchvision==0.6.0。

## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，使用的开源数据集为ImageNet，将数据集上传到服务器任意路径下并解压。 数据集目录结构如下所示：

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
     bash ./test/train_performance_1p.sh --data_path={data/path}	# 1p性能
     ```
     
   - 单机8卡训练
   
     启动8卡训练。
     ```
     bash ./test/train_performance_8p.sh --data_path={data/path}   # 8p性能
     bash ./test/train_full_8p.sh --data_path={data/path}          # 8p精度 完成100个epoch训练大约11h
     bash ./test/train_eval_8p.sh --data_path={data/path}          # 8p验证 
     ```
   - 多机多卡性能数据获取流程
     ```
     1. 安装环境
     2. 开始训练，每个机器所请按下面提示进行配置
            bash ./test/train_performance_multinodes.sh  --data_path=数据集路径 --batch_size=单卡batch_size*所有卡数 --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
     ```
   --data\_path参数填写数据集路径。
   
   对于train_eval_8p.sh，可以在脚本内修改--resume参数，指定evaluate的参数文件路径。
   
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
   --opt_level                         //混合精度等级，默认：'O2'      
   --device                            //使用设备，gpu或npu
   --multiprocessing-distributed       //是否使用多卡训练
   --device_list                       //设备列表，默认：'0,1,2,3,4,5,6,7'
   ```
   

# 训练结果展示

**表 2**  训练结果展示表

|  NAME  | Acc@1  |   FPS   | Epochs | AMP_Type | Torch |
| :----: | :----: | :-----: | :----: | :------: | :---: |
| 1p-NPU |   -    |   319   |   1    |    O2    |  1.5  |
| 1p-NPU |   -    | 747.41  |   1    |    O2    |  1.8  |
| 8p-NPU |   77   |  4127   |  100   |    O2    |  1.5  |
| 8p-NPU | 77.521 | 5306.66 |  100   |    O2    |  1.8  |

# 版本说明

2022.08.01：更新pytorch1.8版本，重新发布。

## 已知问题
无。
