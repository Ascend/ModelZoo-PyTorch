# TextSnake for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

TextSnake提出一种灵活而通用的表征，通过一系列有序、彼此重叠的圆盘（disk）描述文本，每个圆盘位于文本区域的中心轴上，并带有可以变化的半径和方向，可以描述任意形状的场景文本，包括水平文本，多方向文本和曲形文本，并在当时取得了最优或有竞争力的结果。

- 参考实现：

  ```
  url=https://github.com/princewang1994/TextSnake.pytorch
  commit_id=b4ee996d5a4d214ed825350d6b307dd1c31faa07
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

  | Torch_Version | 三方库依赖版本 |
  | :-----------: | :------------: |
  |  PyTorch 1.5  | pillow==8.4.0  |
  |  PyTorch 1.8  | pillow==9.1.0  |

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

- 安装geos(从下面方法中选择一种)

  - 命令安装

    ```
    apt-get install libgeos-dev  # Ubuntu
     
    yum install geos-devel  # centos
    ```

  - 源码安装

    ```
    wget http://download.osgeo.org/geos/geos-3.8.1.tar.bz2
    bunzip2 geos-3.8.1.tar.bz2
    tar xvf geos-3.8.1.tar
    cd geos-3.8.1
    ./configure && make && make install
    ```


## 准备数据集

1. 请用户自行准备好数据集，包含训练集和验证集两部分，可选用的数据集为TotalText。将准备好的数据集上传至源码包根目录下的“data/”文件夹中并解压，解压后训练集和验证集图片分别位于“train/”和“val/”文件夹路径下，该目录下每个文件夹代表一个类别，且同一文件夹下的所有图片都有相同的标签。当前提供的训练脚本中，是以TotalText数据集为例。在使用其他数据集时，修改数据集路径。数据集目录结构参考。

   ```
   ├ data
   ├── total-text
   │    ├──Images ├──Train ──图片1、2、3、4
   │    │         ├──Test  ──图片1、2、3、4
   │    │
   │    ├──gt     ├──Train ──文件1、2、3、4
   │    │         ├──Test  ──文件1、2、3、4
   ```
   
   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。

## 下载预训练模型

请参考原始仓库上的README.md进行预训练模型获取，将下载好的预训练模型放入源码包根目录下新建的`save/synthtext_pretrain`文件夹中。


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

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh
     ```

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --weight_decay                      //权重衰减
   --batch_size                        //训练批次大小
   --momentum                          //动量
   --max_epoch                         //最大训练周期数
   --data_path                         //数据集路径
   --num_workers                       //加载线程数
   --device                            //设备类型
   --world_size                        //分布式训练节点数
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| Precision | Recall | F-measure | FPS  | Npu_nums | Epochs | AMP_Type |
| :-------: | :----: | :-------: | :--: | :------: | :----: | :------: |
|     -     |   -    |     -     |  -   |    1     |   1    |    O1    |
|   0.741   | 0.687  |   0.713   |  29  |    8     |  200   |    O1    |

# 版本说明

## 变更

2023.03.08：更新readme，重新发布。

2021.07.08：首次发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md