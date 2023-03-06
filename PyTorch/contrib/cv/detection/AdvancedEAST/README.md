# AdvancedEAST for Pytorch


-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

AdvancedEast是场景文字检测算法，基于EAST算法，对EAST在长文本检测地方的缺陷进行了重大改进，使长文本预测更加准确。总体来说AdvancedEast检测算法在多角度文字检测方面表现良好，没有明显的缺陷。

- 参考实现：

  ```
  url=https://github.com/BaoWentz/AdvancedEAST-PyTorch
  commit_id=a835c8cedce4ada1bc9580754245183d9f4aaa17
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

  | Torch_Version |             三方库依赖版本              |
  | :-----------: | :-------------------------------------: |
  |  PyTorch 1.5  | torchvision==0.2.2.post3；pillow==8.4.0 |
  |  PyTorch 1.8  |    torchvision==0.9.1；pillow==9.3.0    |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  ```
  > **说明：** 
  > 只需执行一条对应的PyTorch版本依赖安装命令。


## 准备数据集

1. 获取数据集。

   请用户自行获取数据集，分别下载ICPR_text_train_part2_20180313.zip和ICPR_text_train_part1_20180316.zip两个压缩包，在源码包根目录下新建目录icpr和子目录icpr/image_10000、icpr/txt_10000，将压缩包中image_9000、image_1000中的图片文件解压至image_10000中，将压缩包中txt_9000、txt_1000中的标签文件解压至txt_10000中。
   
   数据集目录结构参考如下所示。

   ```
   ├── icpr
         ├──image_10000
              │──图片1
              │──图片2
              │   ...
   
         ├──txt_10000
              │──标注1
              │──标注2
              │   ...
   ```
   
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理（在源码包根目录下执行以下命令）。
   
   ```
   bash test/prep_dataset.sh
   ```


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。依次训练size为256x256，384x384，512x512，640x640，736x736的图片，每个size加载上个size的训练结果，加速模型收敛。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_performance_1p.sh  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_performance_8p.sh  # 8卡性能
     
     bash ./test/train_full_8p.sh  # 8卡精度
     # 默认依次训练256，384，512，640，736五个size，可以指定要训练size，用于恢复中断的训练，例如 `bash test/train_full_8p.sh 640 736`
     ```

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --apex                              //是否使用混合精度
   --device                            //设置训练设备
   --img_path                          //设置图片加载路径
   --epoch_num                         //设置训练周期数 
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表（pytorch1.5)

| Size     | F1-score | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------: | :------:  | :------: | :------: | :------: |
| 256      | -        | 254       | 1        | -        | O1       |
| 256      | -        | 1075      | 8        | 60       | O1       |
| 384      | -        | 118       | 1        | -        | O1       |
| 384      | -        | 680       | 8        | 60       | O1       |
| 512      | -        | 63        | 1        | -        | O1       |
| 512      | -        | 400       | 8        | 60       | O1       |
| 640      | -        | 37        | 1        | -        | O1       |
| 640      | -        | 243       | 8        | 60       | O1       |
| 736      | -        | 34        | 1        | -        | O1       |
| 736      | 62.41%   | 218       | 8        | 60       | O1       |

**表 3**  训练结果展示表（pytorch1.8)

| Size     | F1-score | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------: | :------:  | :------: | :------: | :------: |
| 256      | -        | 306.044   | 1        | -        | O1       |
| 256      | -        | 1876.826  | 8        | 60       | O1       |
| 384      | -        | 147.237   | 1        | -        | O1       |
| 384      | -        | 978.686   | 8        | 60       | O1       |
| 512      | -        | 82.347    | 1        | -        | O1       |
| 512      | -        | 569.184   | 8        | 60       | O1       |
| 640      | -        | 47.418    | 1        | -        | O1       |
| 640      | -        | 361.766   | 8        | 60       | O1       |
| 736      | -        | 38.31     | 1        | -        | O1       |
| 736      | 62.47%   | 273.019   | 8        | 60       | O1       |


# 版本说明

## 变更

2023.03.03：更新readme，重新发布。

2020.08.26：首次发布。

## FAQ

无。

