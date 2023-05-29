# GloRe for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

Glore是一个经典的视频分类网络，其包含了基于图的全局推理网络，他的名称也由此而来，这个特点让Glore可以捕获画面中各个语义之间构成的全局信息，以获取更优的精度。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/GloRe
  commit_id=9c6a7340ebb44a66a3bf1945094fc685fb7b730d
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/video
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.6.0；pillow==8.4.0 |
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

   请用户自行获取原始数据集**UCF-101**，包含训练集和测试集两部分，并在模型源码包根目录`./dataset/UCF101/raw/`路径下新建data文件夹，将获取的数据集上传至该路径下并解压。

   数据集目录结构参考如下所示。
   
   ```
   ├── ucf101
   │   ├── ucf101_{train,val}_split_{1,2,3}_rawframes.txt
   │   ├── ucf101_{train,val}_split_{1,2,3}_videos.txt
   │   ├── annotations
   │   ├── videos
   │   │   ├── ApplyEyeMakeup
   │   │   │   ├── v_ApplyEyeMakeup_g01_c01.avi  
   │   │   ├── YoYo
   │   │   │   ├── v_YoYo_g25_c05.avi
   │   ├── rawframes
   │   │   ├── ApplyEyeMakeup
   │   │   │   ├── v_ApplyEyeMakeup_g01_c01
   │   │   │   │   ├── img_00001.jpg
   │   │   │   │   ├── img_00002.jpg
   │   │   │   │   ├── ...
   │   │   │   │   ├── flow_x_00001.jpg
   │   │   │   │   ├── flow_x_00002.jpg
   │   │   │   │   ├── ...
   │   │   │   │   ├── flow_y_00001.jpg
   │   │   │   │   ├── flow_y_00002.jpg
   │   │   ├── ...
   │   │   ├── YoYo
   │   │   │   ├── v_YoYo_g01_c01
   │   │   │   ├── ...
   │   │   │   ├── v_YoYo_g25_c05     
   ```
   
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。


## 获取预训练模型

请用户自行获取预训练模型**resnet50-lite_3d_8x8_w-glore_2-3_ep-0000.pth**，将获取的预训练模型存放在源码包根目录`./network/pretrained/`路径下。

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
     bash ./test/train_full_1p.sh  # 单卡精度
     
     bash ./test/train_performance_1p.sh  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh  # 8卡精度
     
     bash ./test/train_performance_8p.sh  # 8卡性能
     ```
   
   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --network                           //设置主干网络
   --pretrained                        //加载预训练模型
   --batch-size                        //训练批次大小
   --random-seed                       //随机种子设置
   --backend                           //通信后端
   --amp                               //是否使用混合精度
   --loss-scale                        //混合精度loss scale大小
   --opt-level                         //混合精度类型
   --distributed                       //启动分布式训练
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME | ACC@1    | FPS       | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| 1p-NPU  | -        | 11.647      | 90     | O2       |
| 8p-NPU | 92.39     | 141.31     | 90      | O2       |


# 版本说明

## 变更

2023.03.21：更新readme，重新发布。

2020.07.08：首次发布。

## FAQ

1. 在ARM平台上，若无法使用pip安装0.6.0版本的torchvision，可参考源码readme，进行本地编译安装。
   
   ```
   # 源码链接
   https://github.com/pytorch/vision
   ```
