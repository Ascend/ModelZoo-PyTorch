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

- 安装 detectron2 框架。
  
    ```
    pip3.7 install -e .
    pip3.7 install fvcore --upgrade
    ```

## 准备数据集

1. 获取数据集。

   请用户自行准备**coco**数据集，并将数据集解压后放置在服务器的任意目录下。

   数据集目录结构参考如下所示：

   ```
   |-coco
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
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。


### 获取预训练模型

请用户自行下载预训练模型**R-50.pkl**，上传到服务器任意路径下，并修改源码包根目录下''configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml''文件的**WEIGHTS**参数，指定预训练模型文件的绝对路径，示例如下：

```
WEIGHTS: "/root/txyWorkSpace/Faster_Mask_RCNN_for_PyTorch/R-50.pkl"
```

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
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh --data_path=/data/xxx/
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

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

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2** 训练结果展示表

|  NAME  | Acc@1 |  FPS   | Max_iter | AMP_Type | Torch_Version |
| :----: | :---: | :----: | :------: | :------: | :-----------: |
| 1p-NPU |   -   | 9.644  |   1000   |    O1    |      1.8      |
| 8p-NPU | 37.8  | 66.273 |  90000   |    O1    |      1.8      |

#  版本说明

## 变更

2023.03.02：更新readme，重新发布。

2021.07.14：首次发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md