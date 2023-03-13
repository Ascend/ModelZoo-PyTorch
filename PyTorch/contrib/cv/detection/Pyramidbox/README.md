# Pyramidbox for PyTorch

-   [概述](概述.md)
-   [
  训练环境](
  训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

Pyramidbox提出一种基于anchor的环境辅助方法`PyramidAnchors`，从而引入有监督的信息来为较小的、模糊的和部分遮挡的目标学习环境特征，并设计了`LFPN`，`Context-Sensitive`的架构，更好地融合环境特征和目标特征，从融合特征中更好处理不同尺度的目标。

- 参考实现：

  ```
  url=https://github.com/yxlijun/Pyramidbox.pytorch
  commit_id=76cf3558ef09bf27df15d960f478b7e5b4a6a673
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/detection
  ```

#
训练环境

##
环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.2.2.post3；pillow==8.4.0 |
  | PyTorch 1.8 | torchvision==0.9.1；pillow==9.1.0 |

- 环境
指导。

  请参考《[Pytorch框架训练环境
  ](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  ```
  > **说明:** 只需执行一条对应的PyTorch版本依赖安装命令。

##
数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括WIDER_FACE等，
   在项目根目录下创建WIDER_FACE目录，存放数据集。
   ```
    # $Pyramidbox 是项目根目录
    $Pyramidbox/WIDER_FACE
   ```

   以WIDER_FACE数据集为例，数据集目录结构参考如下所示。

   ```
	|-WIDER_FACE
		|-wider_face_split
			|-wider_face_test.mat
			|-wider_face_test_filelist.txt
			|-wider_face_train.mat
			|-wider_face_train_bbx_gt.txt
			|-wider_face_val.mat
			|-wider_face_val_bbx_gt.txt
		|-WIDER_train
			|-images
				|-0--Parade
				|-1--Handshaking
				...
		|-WIDER_val
			|-images
				|-0--Parade
				|-1--Handshaking
				...
   ```
   > **说明：**
   >该数据集的训练过程脚本只作为一种参考示例。
2. 数据预处理。

	运行prepare_wider_data.py：
	```
	python prepare_wider_data.py --data_path='数据集路径'
	```

## 获取预训练模型
参照原代码仓README下载vgg权重，放在weights目录下。

  ```
    |-Pyramidbox
      |-weights
        |-vgg16_reducedfc.pth
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
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度

     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --batch-size                        //训练批次大小
   --performance                       //是否执行性能模式
   --lr                                //初始学习率
   多卡训练参数：
   --multinpu                          //是否使用多卡训练
   --world_size                        //训练卡数量
   --device-list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | AP | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V |   -   |   -   |   1    |    -     |      1.5      |
| 8p-竞品V |     x |   -  | 100  |    -     |      1.5      |
|  1p-NPU  |   -   |  -  |   1    |    O2    |      1.8      |
|  8p-NPU  |  Easy: 0.9519612346942784;Medium: 0.9446576258551937;Hard: 0.9053749943031708  | xxx  |  100   |    O2    |      1.8      |

# 版本说明

## 变更

2020.10.14：更新内容，重新发布。

2020.07.08：首次发布。

## FAQ

无。