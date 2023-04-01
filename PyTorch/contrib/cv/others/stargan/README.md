# StarGAN for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

最近的研究表明，在两个领域的图像到图像的翻译中取得了显著的成功。然而，现有的方法在处理两个以上的领域时具有有限的可扩展性和稳健性，因为每一对图像领域都要独立建立不同的模型。为了解决这一局限性，我们提出了StarGAN，一种新颖的、可扩展的方法，只用一个模型就可以进行多个领域的图像到图像的翻译。StarGAN的这种统一的模型结构允许在一个网络中同时训练具有不同领域的多个数据集。这导致了StarGAN与现有模型相比具有更高的翻译图像质量，以及灵活地将输入图像翻译到任何所需目标领域的新颖能力。

- 参考实现：

  ```
  url=https://github.com/yunjey/stargan
  commit_id=94dd002e93a2863d9b987a937b85925b80f7a19f
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git	
  code_path=PyTorch/contrib/cv/others
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

   请用户自行下载CelebA数据集。将数据集移动到源码包根目录，并运行“unzip_dataset.sh”脚本。执行后在根目录的“./data”文件夹下生成数据集。

   ```
   bash ./unzip_dataset.sh
   ```

   数据集中包含两部分，一部分是训练图片，即128x128的CelebA人脸照片。另一部分是一个包含特征属性的list_attr_celeba.txt文件。

   数据集目录结构参考如下：

   ```
   ├ data
   ├── celeba
   │    ├──images    
   │    │     ├── image0.jpg …
   │    ├──list_attr_celeba.txt
   ```

   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。

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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  #8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  #8卡性能
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --mode                    //训练或测试，通常使用train 
   --folder_dir              //输出位置，如stargan_NPU_8p    
   --epoch                   //重复训练次数，默认50
   --batch_size              //训练批次大小，默认16
   --distributed             //是否多卡，默认True
   --npus                    //使用多卡的数量，默认1 
   --selected_attrs          //选择的训练特征，默认Black_Hair,Blond_Hair,Brown_Hair,Male,Young
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME | CPU_Type  | Use_Amp       | Acc@1  | FPS       | Epochs   | AMP_Type | Torch_Version |
| :------: | :------:  | :------: | :------:  | :------: | :------: | :------: | :------: |
| 1p-竞品V | X86 | - | - | 62 | 1 | O1 | 1.5 |
| 8p-竞品V | X86 | - | - | 517 | 50 | O1 | 1.5 |
| NPU-1p   | 非Arm | No | -    | 76.9     | 1      | -       | 1.8    |
| NPU-8p | 非Arm | No | -  | 589.801   | 1      | -       | 1.8    |
| NPU-8p | 非Arm | Yes | -  | 656.523   | 1      | O1       | 1.8    |
| NPU-8p | 非Arm | Yes | -  | 714.983   | 1      | O2       | 1.8    |
| NPU-8p | Arm | No | -  | 352.153   | 1      | -       | 1.8    |


# 版本说明

## 变更

2022.02.14：更新readme，重新发布。

2021.07.08：首次发布。

## FAQ

无。