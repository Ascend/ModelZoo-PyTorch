# FixRes for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

  FixRes是一个图像分类网络，该模型使用较低分辨率图像输入对ResNet50网络进行训练，并使用较高分辨率图像输入对训练好的模型进行finetune，最终使用较高分辨率进行测试，以此解决预处理过程中图像增强方法不同引入的偏差。
- 参考实现：
    ```
    url=https://github.com/facebookresearch/FixRes
    commit_id=c9be6acc7a6b32f896e62c28a97c20c2348327d3
    ```

- 适配昇腾 AI 处理器的实现：

    ```
    url=https://gitee.com/ascend/ModelZoo-PyTorch
    code_path=PyTorch/contrib/cv/classification
    ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.2.2.post3 |
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

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，将数据集上传到服务器任意路径下并解压。

   以ImageNet数据集为例，数据集目录结构参考如下所示。

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
   >该数据集的训练过程脚本只作为一种参考示例。


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。在训练之前需要在源码包根目录下创建“train_cache”文件夹，用来存放训练时产生的文件。

   - 单机单卡训练
    
     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能

     bash ./test/finetune_full_1p.sh --data_path=/data/xxx/ --pth_path=real_pre_train_model_path  # 单卡调优
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能

     bash ./test/finetune_full_8p.sh --data_path=/data/xxx/ --pth_path=real_pre_train_model_path  # 8卡调优
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
   --addr                              //主机地址
   --imnet_path                        //数据集路径
   --num_tasks                         //使用卡数 
   --epochs                            //重复训练次数
   --batch                             //训练批次大小
   --learning_rate                     //初始学习率
   --local_rank                        //训练指定用卡
   --pth_path                          //pth文件存放路径  
   ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V |   -   | -  |  train: 120<br>finetune: 60  |    -     |      1.5      |
| 8p-竞品V | - | - | train: 120<br>finetune: 60 |    -     |      1.5      |
|  1p-NPU  |   -   | train: 590<br>finetune: 711  |  train: 120<br>finetune: 60  |    O1    |      1.8      |
|  8p-NPU  |  72.9(72.1 before finetune)  | train: 3318<br>finetune: 3515  |  train: 120<br>finetune: 60   |    O1    |      1.8      |


# 版本说明

## 变更

2023.1.30：更新readme，重新发布。


## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md