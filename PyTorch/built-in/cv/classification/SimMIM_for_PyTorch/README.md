# SimMIM_for_PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

本文档介绍了SimMIM，一个用于掩蔽图像建模的简单框架。通过系统研究，我们发现每个组件的简单设计都显示出非常强大的表示学习性能：使用中等大小的遮罩补丁（例如，32）随机屏蔽输入图像，使文本前任务变得强大；通过直接回归预测RGB值的原始像素并不比复杂设计的斑块分类方法差；预测头可以像线性层一样轻，性能不会比重的更差。
- 参考实现：

  ```
  url=https://github.com/microsoft/SimMIM
  commit_id=519ae7b0999b9d720daa61e3848cd41b8fbd9978
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification
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
    | 硬件    | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
    | 固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
    | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
    | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)|

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip3.7 install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集ImageNet，将数据集上传到服务器任意路径下并解压。

   SimMIM_for_PyTorch使用到的ImageNet数据集目录结构参考如下所示。

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

## Pre-Training

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/pretrain_performance_1p.sh --data_path=real_data_path
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/pretrain_performance_8p.sh --data_path=real_data_path
     bash ./test/pretrain_full_8p.sh --data_path=real_data_path
     ```
     
## Fine-Tuning

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/finetune_performance_1p.sh --data_path=real_data_path 
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/finetune_performance_8p.sh --data_path=real_data_path
     bash ./test/finetune_full_8p.sh --data_path=real_data_path
     ```
   --data_path参数填写数据集路径。

   
   
   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data-path                              //数据集路径
   --output                                 //输出文件路径
   --batch-size                             //批大小
   --local_rank                             //使用的NPU的id
   --pretrained                             //预训练模型路径
   --opts TRAIN.EPOCHS                      //重复训练次数
   ```

# 训练结果展示

## Pre-Training训练结果展示表
| NAME  | Acc@1  | FPS  | Epochs  | AMP_Type  | Torch  |
|---|---|---|---|---|---|
| 1p-NPU  |   | 198  | 2  | O1  | 1.8  |
| 1p-竞品  |   | 185.6  | 2  | O1  | 1.8  |
| 8p-NPU  |   | 1555  | 100  | O1  | 1.8  |
| 8p-竞品   |   | 1422 | 100  | O1  | 1.8  |

## Fine-Tuning训练结果展示表
| NAME  | Acc@1  | FPS  | Epochs  | AMP_Type  | Torch  |
|---|---|---|---|---|---|
| 1p-NPU  |   | 194  | 2  | O1  | 1.8  |
| 1p-竞品  |   | 189.5  | 2  | O1  | 1.8  |
| 8p-NPU  | 81.782  | 1513  | 100  | O1  | 1.8  |
| 8p-竞品   | 82.06  | 1410  | 100  | O1  | 1.8  |
# 版本说明

## 变更

2022.12.14：首次发布。

## 已知问题

无。
