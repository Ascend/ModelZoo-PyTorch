# FOTS for PyTorch

- [概述](https://gitee.com/ascend/docs-openmind/blob/master/guide/modelzoo/pytorch_model/tutorials/概述.md)
- [准备训练环境](https://gitee.com/ascend/docs-openmind/blob/master/guide/modelzoo/pytorch_model/tutorials/准备训练环境.md)
- [开始训练](https://gitee.com/ascend/docs-openmind/blob/master/guide/modelzoo/pytorch_model/tutorials/开始训练.md)
- [训练结果展示](https://gitee.com/ascend/docs-openmind/blob/master/guide/modelzoo/pytorch_model/tutorials/训练结果展示.md)
- [版本说明](https://gitee.com/ascend/docs-openmind/blob/master/guide/modelzoo/pytorch_model/tutorials/版本说明.md)

# 概述

## 简述

随机场景文本发现被认为是文档分析社区中最困难和最有价值的挑战之一。大多数现有方法将文本检测和识别视为单独的任务。端到端可训练的Fast Oriented Text Spotting（FOTS）网络，可用于同时进行检测和识别，在两个互补任务之间共享计算和视觉信息。特别地，引入了RoIRotate以在检测和识别之间共享卷积特征。受益于卷积共享策略，与基线文本检测网络相比，FOTS具有很少的计算开销，并且联合训练方法学习更多的通用特征，使得FOTS的方法比这两个阶段的方法更好。在ICDAR 2015、ICDAR 2017 MLT和ICDAR 2013数据集上进行的实验表明，提出的方法明显优于最新方法。

- 参考实现：

  ```
  url=https://github.com/Wovchena/text-detection-fots.pytorch
  commit_id=04b969c87491630dce38cdb48932aa33115798c6
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/detection
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  ```
  pip install -r requirements.txt
  ```
  > **说明：** 
  >只有opencv-python==3.4.3.18版本时，模型收敛，其他版本不收敛！


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括SynthText、ICDAR2015等，其中SynthText用于pretrain，ICDAR2015用于finetune和test，将数据集上传到服务器任意路径下并解压。

   以ICDAR2015、SynthText数据集为例，数据集目录结构参考如下所示。

   ```
   ├── SynthText 
   │    ├──类别1├──序号1──图片1、2、3、4 
   │    │      │  ...         
   │    │      ├──序号xx──图片1、2、3、4 
   │    │                      
   │    ├──类别2├──序号1──图片1、2、3、4 
   |           |  ...
   |           ├──序号xx──图片1、2、3、4
   
   ├── ICDAR2015
   │    ├──ch4_training_vocabularies_per_image├──voc_img_1.txt
   │    │                                     │  ...         
   │    │                                     ├──voc_img_xxx.txt 
   │    │                                      
   │    ├──ch4_training_localization_transcription_gt├──gt_img_1.txt 
   │    │                                            │  ...
   │    │                                            ├──gt_img_xxx.txt
   │    │                      
   │    ├──ch4_training_images├──img_1.jpg 
   │    │                     │  ...           
   │    │                     ├──img_xxx.jpg 
   │    │                      
   │    ├──ch4_training_vocabulary.txt
   │    │  
   |    ├──ch4_test_images
   │    |    ├──img_1.jpg
   │    │    |  ...
   │    │    │──img_xxx.jpg
   │    │
   |    ├──gt.zip                           
   ```

   > **说明：** 该数据集的训练过程脚本只作为一种参考示例。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练精度脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_pretrain_1p.sh --data_path=/data/xxx/SynthText
     bash ./test/train_full_finetune_1p.sh --data_path=/data/xxx/ICDAR2015 
     bash ./test/test_1p.sh --data_path=/data/xxx/ICDAR2015
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_pretrain_8p.sh --data_path=/data/xxx/SynthText
     bash ./test/train_full_finetune_8p.sh --data_path=/data/xxx/ICDAR2015 
     bash ./test/test_8p.sh --data_path=/data/xxx/ICDAR2015
     ```

   --data_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                         //数据集路径
   --batch-size                        //训练批次大小
   --batches-before-train              //更新梯度的频率
   --num-workers                       //加载数据进程数      
   --continue-training                 //finetune
   --pf                                //performance
   --output-folder                     //模型的输出路径
   --checkpoint                        //训练好的模型权重路径
   --height-size                       //调整输入图片的高度
   --dis                               //是否多卡
   ```

   训练完成后，权重文件保存在./runs/路径下，并输出模型性能信息和精度信息。

3. 运行训练性能脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/ICDAR2015
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/ICDAR2015
     ```

   --data_path参数填写数据集路径。
   
   模型评估脚本参数说明如下。
   
   ```
   公共参数：
   --data_path                         //数据集路径
   --batch-size                        //训练批次大小
   --batches-before-train              //更新梯度的频率
   --num-workers                       //加载数据进程数      
   --continue-training                 //finetune
   --pf                                //performance
   ```
   
   运行完成后，获得模型性能信息。

# 训练结果展示

**表 2** 训练结果展示表

|  NAME  | Hmean  |   FPS   | Epochs | AMP_Type | Torch_Version |
| :----: | :----: | :-----: | :----: | :------: | :-----------: |
| 1p-NPU |   -    | 21.263  |   20   |    O2    |      1.8      |
| 8p-NPU | 80.796 | 54.3062 |  583   |    O2    |      1.8      |

# 版本说明

## 变更

2023.03.02：更新readme，重新发布。

2022.09.16：首次发布。

## FAQ

无。