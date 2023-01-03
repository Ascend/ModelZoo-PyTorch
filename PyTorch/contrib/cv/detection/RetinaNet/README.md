# RetinaNet(Detectron2) for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

## 概述

### 简述

RetinaNet算法源自《Focal Loss for Dense Object Detection》，该论文最大的贡献在于提出了Focal Loss用于解决one-stage算法中正负样本的比例严重失衡问题，从而创造了RetinaNet（One Stage目标检测算法）这个精度超越经典Two Stage的Faster-RCNN的目标检测网络。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/retinanet.py
  commit_id=133d22ed09bff7de9eb25c1a04fd7bd87b8d8879
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/detection
  ```

- 通过Git获取代码方法如下：

  ```
  git clone {url}        # 克隆仓库的代码   
  cd {code_path}         # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

- 通过单击“立即下载”，下载源码包。

#  准备训练环境

### 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1** 版本配套表

  | 配套          | 版本                                                         |
  | ------------- | ------------------------------------------------------------ |
  | 硬件          | [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | NPU固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial ) |
  | CANN          | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch       | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fptes)》。

- 安装依赖。

  ```
  cd /${模型文件夹名称}  
  pip install -r requirements.txt
  ```

- 安装 detectron2 框架。
  
    ```
    pip3.7 install -e .
    pip3.7 install fvcore --upgrade
    ```

### 准备数据集

- 请用户自行准备[coco]( http://cocodataset.org/#home)数据集，并将数据集解压后放置在服务器的任意目录下。

    - 数据集目录结构如下：

      ```
      |-- annotations
      |   |-- captions_train2017.json
      |   |-- captions_val2017.json
      |   |-- instances_train2017.json
      |   |-- instances_val2017.json
      |   |-- person_keypoints_train2017.json
      |   |-- person_keypoints_val2017.json
      |-- train2017
      |-- val2017
      |-- test2017
      |   |-- 000000000001.jpg
      |   |-- 000000000016.jpg
      ```

### 获取预训练模型

将预训练模型[R-50.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl)放在

```
/root/txyWorkSpace/Faster_Mask_RCNN_for_PyTorch/
```

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。	

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

- 单机单卡训练

  启动单卡训练。

  ```
  bash ./test/train_full_1p.sh --data_path=数据集路径    
  ```

  测试单卡性能

  ```
  bash ./test/train_performance_1p.sh --data_path=数据集路径  
  ```

- 单机8卡训练

  启动8卡训练。

  ```
  bash ./test/train_full_8p.sh --data_path=数据集路径 
  ```

- 测试8卡性能

  ```
  bash ./test/train_performance_8p.sh --data_path=数据集路径 
  ```

- 8卡评估

  ```
  bash ./test/train_eval_8p.sh --data_path=数据集路径
  ```

--data_path参数填写数据集路径。

模型训练脚本参数说明如下。

```
公共参数：
--config-file                       //模型配置文件
--device_id 0,1,2,3,4,5,6,7         //指定训练用卡
--num-gpus                          //指定训练卡数
--batch-size                        //训练过程批量大小
AMP 1                               //是否开启混合精度训练，1代表开启，0代表关闭
SOLVER.BASE_LR                      //基础学习率
SOLVER.MAX_ITER                     //最高迭代次数
```

训练完成后，权重文件默认会写入到和test文件同一目录下，并输出模型训练精度和性能信息到网络脚本test下output文件夹内。

# 训练结果展示

**表 2** 训练结果展示表

| 名称   | 精度 | 性能      | torch版本 |
| ------ | ---- | --------- | --------- |
| NPU-1P | -    | 5.58fps   | 1.5       |
| NPU-8P | 37.2 | 34.875fps | 1.5       |
| NPU-1P | -    | 9.644fps  | 1.8       |
| NPU-8P | 37.8 | 66.273fps | 1.8       |

#  版本说明

## 变更

2022.07.14：更新torch1.8版本，重新发布。

2021.07.14：首次发布。

## 已知问题

无。