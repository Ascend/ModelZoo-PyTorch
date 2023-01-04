# BMN for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

时间动作建议生成是一项具有挑战性和前景的任务，旨在定位真实世界视频中可能发生动作或事件的时间区域。BMN利用边界匹配（BM）机制来评估密集分布提案的置信度分数，该机制将提案作为起始和结束边界的匹配对，并将所有密集分布的BM对组合到BM置信图中，该方法同时生成具有精确时间边界和可靠置信分数的提名。

- 参考实现：

  ```
  url=https://github.com/JJBOY/BMN-Boundary-Matching-Network
  commit_id=a92c1d79c19d88b1d57b5abfae5a0be33f3002eb
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/video
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
  | 硬件 | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | NPU固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)|

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

  数据集原本是ActivityNet，不过这个数据集非常庞大，实际复现时使用已经提取好的特征数据集，这个数据集可以从github原仓(https://github.com/JJBOY/BMN-Boundary-Matching-Network) 上找到下载方式。

  数据集目录结构参考如下所示。

  ```
   ├── csv_mean_100
         ├── 视频1的特征csv
         ├── 视频2的特征csv
         │   ...             
         ├── 视频19228的特征csv
  ```

  该数据集无需其他预处理。

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
     #test/train_full_1p，单p上训练10个epoch，运行时间2-3小时,输出精度日志./output/0/train_full.log
     bash ./test/train_full_1p.sh --data_path=/data/xxx/
     
     #test/train_performance_1p，单p上训练1个epoch，运行时间约10余分钟,输出性能日志./output/0/train_perf.log
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     #test/train_full_8p，8p上训练10个epoch，运行时间大约为0.5小时,输出精度日志./output/8/train_full.log
     bash ./test/train_full_8p.sh --data_path=/data/xxx/   

     #test/train_performance_8p，8p上训练1个epoch，运行时间约2分钟,输出性能日志./output/8/train_perf.log
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/    
     ```
   - 验证

     ```
     本项目的验证和训练过程已经合并，无需单独进行验证。
     ```

   --data_path参数填写数据集路径，请用户根据实际修改。

   模型训练脚本参数说明如下。

   ```
   主要参数：

   --mode                              //训练模式
   --finetune                          //是否微调
   --training_lr                       //学习率
   --weight_decay                      //权重衰减
   --train_epochs                      //训练轮数
   --video_info                        //数据集信息
   --video_anno                        //数据集注释
   --data_path                         //数据路径
   --result_file                       //输出位置
   --save_path_fig                     //存储位置
   --is_distributed                    //是否分布式训练
   --DeviceID                          //使用NPU编号
   --world_size                        //使用NPU总数
   --loss_scale                        //精度类型
   --opt-level                         //混合精度类型
   多卡训练参数：
   ----nproc_per_node                  //训练使用卡的数量

   ```

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@100 |  FPS | Epochs | AMP_Type | PyTorch版本 |
| ------- | ------- | ---: | ------ | -------: | ------- |
| 1p-NPU | 75.11 |  65 | 10      |        O1 |1.5    |
| 1p-NPU  | 75 |  58.96 | 10      |       O1 |1.8    |
| 8p-NPU | 74.82 | 553 | 10    |        O1 |1.5    |
| 8p-NPU  | 75 | 525.94 | 10    |       O1 |1.8    |


# 版本说明

## 变更

2022.12.15：更新torch1.8版本，重新发布。

2022.02.14：首次发布。

## 已知问题

无。

