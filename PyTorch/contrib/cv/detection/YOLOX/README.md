# YOLOX

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述
YOLOX对YOLO系列进行了一些有经验的改进，将YOLO检测器转换为无锚方式，并进行其他先进的检测技术，即解耦头和领先的标签分配策略SimOTA，在大规模的模型范围内获得最先进的结果。

- 参考实现：

  ```
  url=https://github.com/Megvii-BaseDetection/YOLOX
  commit_id=dd5700c24693e1852b55ce0cb170342c19943d8b
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
  | 硬件       | [1.0.15.3](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [22.0.0.3](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC1.1](https://www.hiascend.com/software/cann/commercial?version=5.1.RC1.1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  pip install -v -e .
  pip install cython 
  pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
  ```

## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，将数据集上传到服务器任意路径下并解压。

   以ImageNet2012数据集为例，数据集目录结构参考如下所示。

   ```
   ├── coco2017
   │   ├── annotations
   │          ├── captions_train2017.json
   │          ├── captions_val2017.json
   │          ├── instances_train2017.json
   │          ├── instances_val2017.json
   │          ├── person_keypoints_train2017.json
   │          ├── person_keypoints_val2017.json
   │   ├── train2017
   │          ├── 000000000009.jpg
   │          ├── 000000000025.jpg
   │          ├── ......
   │   ├── val2017
   │          ├── 000000000139.jpg
   │          ├── 000000000285.jpg        
   |          ├── ......          
   ```
   
   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。

# 开始训练

## 训练模型

模型存在动态Shape，为提升性能，固定shape使用分档策略，同时模型默认开启多尺度，故训练前期持续有算子编译，现象为iter_time抖动。性能数据请关注性能稳定之后（400step之后很少有算子编译）。精度测试须300epoch，前期性能波动对整体影响较小。

shell脚本会将传入的`data_path`软连接到`./datasets`目录下，默认使用VOC2012数据集，使用其它数据集须自行修改配置文件并将数据转为COCO格式。

注：压测后发现模型对随机种子敏感，使用不同种子最终精度会有明显抖动，甚至会有低概率mAP有20%以上抖动（竞品上有类似现象）。当前针对默认配置（VOC2012/yolox-s）固定了随机种子，保证结果可复现，若更换了模型配置或数据集，请自行修改相关参数。随机种子设置在`yolox/exp/base_exp.py`中设置。

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/
     ```

   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --name                              //模型名称
   --devices                           //训练设备
   --dist-backend                      //通信后端
   --batch-size                        //训练批次大小
   --dist-url                          //启用分布式训练网址
   --world-size                        //分布式训练节点数量
   --num_machines                      //训练节点数
   --device_id                         //训练单卡id
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

默认配置（yolox-s + VOC2012）

| 名称   | 精度 | 性能    |
| ------ | ---- | ------- |
| A40-1p | -    | 37.62 fps |
| 910A-1p | -    | 54.05 fps |
| A40-8p | 0.410 | 285.85 fps|
| 910A-8p | 0.407 | 320 fps  |

可选配置（yolox-x + COCO 众智交付）

| 名称   | 精度 | 性能    |
| ------ | ---- | ------- |
| V00-1p | -    | 20 fps   |
| 910A-1p | -    | 20.5 fps |
| V100-8p | 50.7 | 106 fps  |
| 910A-8p | 50.5 | 140 fps  |

# 版本说明

## 变更

2022.07.27：更新pytorch1.8版本，重新发布。

2021.07.08：首次发布。

## 已知问题

无。

