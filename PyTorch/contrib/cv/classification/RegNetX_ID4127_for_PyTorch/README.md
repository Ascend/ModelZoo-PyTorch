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
  | 硬件 | [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)|

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```
  注意:如果无法直接安装对应版本的torchvision，可以使用源代码安装对应版本。源代码参考链接:https://github.com/pytorch/vision

## 准备数据集

1. 获取数据集。

   下载开源数据集包括ImageNet，将数据集上传到服务器任意路径下并解压。
   然后用[下列脚本](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)将验证图像移动到标记的子文件夹
    
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
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/   

     bash ./test/train_performance_8p.sh --data_path=/data/xxx/
     ```
   - 微调模型
     ```
     # finetuning
     bash ./test/train_finetune_1p.sh --data_path=real_data_path --pth_path=./checkpoints/checkpoint.pth.tar
     ```

   --data_path：数据集路径

   --pth_path：模型权重路径

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data                         //数据集路径
   --train-batch                   //训练批次大小
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
   
   Log path:

    test/output/devie_id/train_${device_id}.log           # training detail log
    
    test/output/devie_id/RegNetX_8p_perf.log  # 8p training performance result log
    
    test/output/devie_id/RegNetX_8p_acc.log   # 8p training accuracy result log。

# 训练结果展示

**表 2**  训练结果展示表


| NAME    | Acc@1  |    FPS  | Epochs | AMP_Type | PyTorch版本 |
| ------- | -----  |   ---:  | ------ | -------: |  -------   |
| 1p-NPU  |   -    |   117  | 1      |    O2    |   1.5    |
| 1p-NPU  |   -    |   1724.3   | 1      |    O2    |   1.8    |
| 8p-NPU  | 77.167   | 7460     |  100   | O2    |    1.5   |
| 8p-NPU  | 77.34    |  7132.5    |  100   |  O2   |    1.8   |

# 版本说明

## 变更

2022.10.24：更新torch1.8版本，重新发布。

2020.12.19：首次发布。

## 已知问题

无。

