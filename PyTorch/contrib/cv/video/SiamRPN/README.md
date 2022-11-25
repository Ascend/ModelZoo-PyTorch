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

- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1** 版本配套表

  | 配套          | 版本                                                         |
  | ------------- | ------------------------------------------------------------ |
  | 硬件          | [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | NPU固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN          | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2 ) |
  | PyTorch       | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fptes)》。

- 安装依赖。

  ```
  pip install -r requirements.txt
  ```

## 准备数据集

1. **获取数据集预处理所需要的脚本。**

   将原始[GPU仓](https://github.com/STVIR/pysot/tree/master/training_dataset)下的coco, det, vid以及yt_bb目录文件下载到本工程下的./training_dataset目录下。        

2. **获取数据集(训练集)。**

   用户自行获取原始数据集，可选用的开源数据集包括COCO, DET, VID以及Youtube-bb等，按照[链接](https://github.com/STVIR/pysot/tree/master/training_dataset)下载和处理四个数据集。

   以[COCO数据集](https://github.com/STVIR/pysot/tree/master/training_dataset/coco)为例，参考[readme.md](https://github.com/STVIR/pysot/blob/master/training_dataset/coco/readme.md)获取数据集，处理前的数据集目录结构参考如下所示：
   
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
   
   coco数据集处理完成后的文件目录结构：
   
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
   
   依次处理上述4个数据集，并将处理后的数据集放在创建的SiamRPN/training_dataset目录下，目录结构参考如下所示：
   
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
   
   文件目录与SiamRPN/pysot-master/experiments/siamrpn_r50_l234_dwxcorr_8gpu/config.yaml中保持一致即可。
   
3. **数据集下载（测试集)--VOT2016**

   下载[VOT-2016数据集](http://votchallenge.net)并放在创建的SiamRPN/testing_dataset路径下。同时也能通过[json文件](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA#list/path=%2F)下载该数据集，目录结构如下：

   ```
    ├── SiamRPN
       ├── testing_dataset
          ├── VOT2016
             ├──类别1
             ├──类别2
             .....
   ```
   

## 获取预训练模型

**1.配置文件获取**：将[代码链接](https://github.com/STVIR/pysot)中experiments中的文件下载到本工程中新创建的./pysot-master/experiments/路径下，本工程使用的配置文件为[原始GPU仓](https://github.com/STVIR/pysot)下的./pysot/experiments/siamrpn_r50_l234_dwxcorr_8gpu/config.yaml文件。

**2.预训练权重获取**：请参考原始仓库上的[MODEL_ZOO.md](https://github.com/STVIR/pysot/blob/master/MODEL_ZOO.md)进行预训练模型获取。下载[siamrpn_r50_l234_dwxcorr](https://drive.google.com/open?id=1Q4-1563iPwV6wSf_lBHDj5CPFiGSlEPG)所对应Model的预训练模型model.pth放至在源码包根目录下的新路径./pretrained_models下并重命名为resnet50.model。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}/pysot-master 
   ```

2. 进入配置好依赖项的环境下（可选）。

   ```
   conda activate XXX    # XXX为创建的虚拟环境，依赖库等必须完整
   ```

3. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - **单机单卡训练**

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh 
     ```

   - **单机8卡训练**

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh   
     ```

   - **单机单卡性能**

   - 启动单卡性能。

     ```
     bash ./test/train_performance_1p.sh
     ```

   - **单机8卡性能**

   - 单机8卡性能。

     ```
     bash ./test/train_performance_8p.sh
     ```

   - **网络训练过程权重保存路径：**

     单卡训练完成后，权重文件保存在SiamRPN/pysot-master/snapshot_1p，8P的权重文件保存在SiamRPN/pysot-master/snapshot_8p下，并输出模型训练精度和性能信息。

   - **日志保存路径：**

     ```
     Log path:
     
     	test/output/devie_id/train_**${**device_id**}**.log           # training detail log
     
     	test/output/devie_id/train_per_**${**device_id**}**.log   # training performance result log
     
     	test/output/devie_id/SiamRPN_for_PyTorch_**${**RANK_SIZE**}**p_acc.log   # training accuracy result log
     ```
     
     

   # 训练结果展示

   **表 2** 训练结果展示表
   
   |    NAME    | FPS↑  | Accuracy↑ | Robustness**↓** | EAO↑  | Torch_version |
   | :--------: | :---: | :-------: | :-------------: | :---: | :-----------: |
   |   1p-NPU   | 36.89 |     -     |        -        |   -   |      1.5      |
   |   1p-NPU   |  49   |     -     |        -        |   -   |      1.8      |
   | 8p-e19-NPU | 203.7 |   0.636   |      0.228      | 0.429 |      1.5      |
   | 8p-e19-NPU | 280.1 |   0.650   |      0.224      | 0.431 |      1.8      |
   | 8P-e20-NPU | 203.7 |   0.638   |      0.247      | 0.405 |      1.5      |
   | 8p-e20-NPU | 280.1 |   0.655   |      0.252      | 0.406 |      1.8      |
   
   # 版本说明

   ## 变更

   2022.11.16：更新内容，重新发布。

   2020.11.10：首次发布。

   ## 已知问题

   无。
