# CenterNet for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述

## 简述

CenterNet使用关键点检测的方法去预测目标边框的中心点，然后回归出目标的其他属性，例如大小、3D位置、方向甚至是其姿态。而且这个方向相比之前的目标检测器，实现起来更加简单，推理速度更快，精度更高。

- 参考实现：

  ```
  url=url=https://github.com/xingyizhou/CenterNet.git 
  commit_id=5b1a490a52da57d3580e80b8bb4bbead9ef2af96
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/detection
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
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```
  pip install -r requirements.txt
  ```

- 安装COCOAPI

  ```
  git clone https://github.com/cocodataset/cocoapi.git
  cd cocoapi/PythonAPI
  python setup.py install
  ```

- 编译可变形卷积（来自DCNv2）

  ```
  cd ./src/lib/models/networks/DCNv2
  ./make.sh
  ```

- 编译NMS

  ```
  cd ./src/lib/external
  make
  ```

## 准备数据集

1. 获取数据集。

   用户自行获取Pscal VOC数据集，将数据集上传到服务器任意路径下并解压。

     - 运行脚本

       ~~~
       cd ./src/tools/
       bash get_pascal_voc.sh
       ~~~

     - 上述脚本内容包含:

       - 从VOC网站下载、解压缩和移动Pascal VOC图像。
       - 下载COCO格式的Pascal VOC注释（从Detectron下载）。
       - 将train/val 2007/2012注释文件合并到单个json中。

   数据集目录结构参考如下所示。

   ```
   |-- data
   `-- |-- voc
       `-- |-- annotations
           |   |-- pascal_trainval0712.json
           |   |-- pascal_test2017.json
           |-- images
           |   |-- 000001.jpg
           |   ......
           `-- VOCdevkit        
   ```

   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。 

备注：Vocdevkit需要用[faster rcnn](https://github.com/rbgirshick/py-faster-rcnn/blob/master/tools/reval.py)去运行评估脚本。

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
     bash ./test/train_full_1p.sh --data_path=数据集路径    
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=数据集路径 
     ```

   --data\_path参数填写数据集路径。

   3.运行测试脚本。

   ```
   bash ./test/train_eval.sh --data_path=数据集路径
   ```

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                         //数据集路径
   --num_workers                       //加载数据进程数      
   --num_epochs                        //重复训练次数
   --batch_size                        //训练批次大小
   --lr                           		 //初始学习率，默认：3.54e-4
   --device_list                       //训练指定训练用卡
   --world-size                        //分布式训练节点数量
   ```

训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 | FPS   | Epochs | AMP_Type | Torch_version |
| ------- | ----- | :---- | ------ | :------- | ------------- |
| 1p-竞品 | -     | -     | -      | -        | -             |
| 8p-竞品 | -     | -     | -      | -        | -             |
| 1p-NPU  | -     | 91    | 1      | O1       | 1.5           |
| 1p-NPU  | -     | 95.44 | 1      | O1       | 1.8           |
| 8p-NPU  | 71.16 | 657   | 90     | O1       | 1.5           |
| 8p-NPU  | 71.12 | 840.208| 90     | O1       | 1.8           |

备注：OBS无模型归档数据，以Torch1.5版本的自验结果作为标杆数据。

# 版本说明

## 变更

2022.09.20：更新pytorch1.8版本。

2021.10.09：首次发布。

## 已知问题

1. 若出现无法找到datasets包的问题，本模型使用的是lib目录下的本地文件，请删除环境中同名三方库

