# SiamRPN++ for PyTorch

- [概述](https://gitee.com/ascend/docs-openmind/blob/master/guide/modelzoo/pytorch_model/tutorials/概述.md)
- [准备训练环境](https://gitee.com/ascend/docs-openmind/blob/master/guide/modelzoo/pytorch_model/tutorials/准备训练环境.md)
- [开始训练](https://gitee.com/ascend/docs-openmind/blob/master/guide/modelzoo/pytorch_model/tutorials/开始训练.md)
- [训练结果展示](https://gitee.com/ascend/docs-openmind/blob/master/guide/modelzoo/pytorch_model/tutorials/训练结果展示.md)
- [版本说明](https://gitee.com/ascend/docs-openmind/blob/master/guide/modelzoo/pytorch_model/tutorials/版本说明.md)

# 概述

## 简述

SiamRPN++是一个由ResNet架构驱动的Siam跟踪器，其内部采用一种简单而有效的采样策略来打破空间不变性的限制。网络中使用一种分层的特征聚合结构用于互相关操作，这有助于跟踪器根据在多个层次上学习到的特征预测相似度图。作为一种高效的视觉跟踪模型，该模型在跟踪精度方面达到了新的水平，同时以35帧/秒的速度高效运行。

- 参考实现：

  ```
  url=https://github.com/STVIR/pysot
  commit_id=9b07c521fd370ba38d35f35f76b275156564a681
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https:https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/video
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


## 准备数据集

1. **获取数据集预处理所需要的脚本。**

   将原始仓（源码实现）`pysot/training_dataset/`目录下的coco，det，vid以及yt_bb目录文件下载到本工程下的`./training_dataset`目录下。        

2. **获取数据集(训练集)。**

   请用户自行获取原始数据集，可选用的开源数据集包括COCO，DET，VID以及Youtube-bb等，获取方式可参原始仓（源码实现）`pysot/training_dataset/`路径下的coco，det，vid以及yt_bb目录下的readme。
   
   以COCO数据集为例，处理前的数据集目录结构参考如下所示。

   ```
   ├── coco
         ├──train2017
                 │──图片1
                 │──图片2
                 │──图片3
                 │   ...       
         ├──val2017  
                 │──图片1
                 │──图片2
                 │──图片3
                 │   ... 
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。
   
2. **进入下载完成的数据集目录下。**

   ``` 
   cd /${模型文件夹名称}/training_dataset/coco
   ```

3. **数据预处理（以COCO数据集为例）。**

   将下载好的数据集裁剪成511*511分辨率并生成对应的train.json文件。

   - 预处理第一步（裁剪至相应尺寸）。
   
     ```
     python par_crop.py 511 12
     ```
   
   - 预处理第二步（生成对应的train.json文件）。
   
     ```
     python gen_json.py
     ```
   
   coco数据集处理完成后的文件目录结构参考如下所示。
   
   ```
   ├──coco
       ├──crop511
           ├──train2017
              ├──类别1
                  │──图片1
                  │──图片2
                  │   ...       
              ├──类别2
                  │──图片1
                  │──图片2
                  │   ... 
           ├──val2017
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

   依次处理上述4个数据集，并将处理后的数据集放在`SiamRPN/training_dataset`目录下，目录结构参考如下所示。
   
   ```
   ├── SiamRPN
   	├── training_dataset
            ├──coco
                 ├──crop511
                 ├──train2017.json
                 ├──gen_json.py
                 ├──par_crop.py
                 .....
            ├──det
                 ├──crop511
                 ├──train.json
                 ├──gen_json.py
                 ├──par_crop.py
                 .....
            ├──vid
                 ├──imagenet2015
                     ├──crop511
                     ├──train.json
                     ├──gen_json.py
                     ├──par_crop.py
                     .....
            ├──yt_bb
                 ├──crop511
                 ├──train.json
                 ├──gen_json.py
                 ├──par_crop.py
                 .....
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

3. **数据集下载（测试集)--VOT2016**

   请用户自行下载VOT-2016数据集，下载方式可参考原始仓（源码实现）`pysot/testing_dataset/`目录下的readme，将下载好的测试集放在`SiamRPN/testing_dataset`路径下，目录结构参考如下。

   ```
    ├── SiamRPN
       ├── testing_dataset
          ├── VOT2016
             ├──类别1
             ├──类别2
             .....
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。


## 获取预训练模型

**1.配置文件获取**：将原始仓（源码实现）中`experiments/`目录下的文件下载到本工程中的`./pysot-master/experiments`/路径下，本工程使用的配置文件为原始仓中的`./pysot/experiments/siamrpn_r50_l234_dwxcorr_8gpu/config.yaml`文件。

**2.预训练权重获取**：请参考原始仓（源码实现）readme进行预训练模型获取，下载siamrpn_r50_l234_dwxcorr对应模型的预训练权重model.pth，存放在源码包根目录下的`./pretrained_models`下，并重命名为resnet50.model。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}/pysot-master 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh  # 单卡精度
     
     bash ./test/train_performance_1p.sh  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh  # 8卡精度
     
     bash ./test/train_performance_8p.sh  # 8卡性能
     ```

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh
     ```

   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --seed                              //随机数种子设置
   --cfg                               //参数配置
   --is_performance                    //设置是否进行性能测试
   --max_step                          //设置最大的迭代数
   ```

   单卡训练完成后，权重文件保存在SiamRPN/pysot-master/snapshot_1p，8P的权重文件保存在SiamRPN/pysot-master/snapshot_8p下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2** 训练结果展示表

|    NAME    | FPS↑  | Accuracy↑ | Robustness**↓** | EAO↑  | AMP_Type | Torch_Version |
| :--------: | :---: | :----: | :------: | :---: | :-----: | :-----: |
|   1p-NPU   |  49   |     -     |        -        |   -   |O1   |1.8   |
| 8p-e19-NPU | 280.1 |   0.650   |      0.224      | 0.431 | O1 | 1.8  |
| 8p-e20-NPU | 280.1 |   0.655   |      0.252      | 0.406 |  O1 |    1.8 |

   # 版本说明

   ## 变更

   2023.03.13：更新readme，重新发布。

   2020.11.10：首次发布。

   ## FAQ

   无。
