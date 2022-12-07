# SSD_MobileNetV1 for PyTorch 

* [概述](概述.md)
* [准备训练环境](开始训练.md)
* [开始训练](开始训练.md)
* [版本说明](版本说明.md)

## 概述

### 简述

MobileNetV1是基于深度级可分离卷积构建的网络。 MobileNetV1将标准卷积拆分为了两个操作：深度卷积和逐点卷积 。
SSD是一种one-stage的目标检测框架。SSD_MobileNetV1使用MobileNetV1提取有效特征，之后SSD通过得到的特征图的信息进行检测。

* 参考实现：

      url=https://github.com/qfgaohao/pytorch-ssd
      commit_id=f61ab424d09bf3d4bb3925693579ac0a92541b0d

* 适配昇腾 AI 处理器的实现：

      url=https://gitee.com/ascend/ModelZoo-PyTorch.git
      code_path=PyTorch/contrib/cv/detection

* 通过Git获取代码方法如下：
* 
      git clone {url}       # 克隆仓库的代码    
      cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
* 通过单击“立即下载”，下载源码包。

## 准备训练环境

### 准备环境

* 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表1** 版本配套表

  | **配套**	 |   **版本**    |
  |:--------:|:-------------:|
  |硬件版本| [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)  |
  |NPU固件与驱动|[5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial )|
  |CANN | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  |PyTorch| [ 1.8.1]( https://gitee.com/ascend/pytorch/tree/master/)  |
	
* 环境准备指导。

	请参考：《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes/ptes_00001.html)》。

* 安装依赖。

	    pip install -r requirements.txt
### 获取预训练模型
* 在源码包根目录下新建"models/"文件夹，[下载所需的预训练模型](https://storage.googleapis.com/models-hao/mobilenet_v1_with_relu_69_5.pth)并将预训练模型放置在"models/"文件夹下。
### 准备数据集

1.获取数据集。

用户自行获取原始数据集，可选用的开源数据集包括VOCdevkit等，将数据集上传到服务器任意路径下并解压。

以VOCdevkit数据集为例，数据集目录结构参考如下所示。

	|——VOCdevkit
		|——VOC2007（VOC2012）
			|——Annotations
			|——ImageSets
			|——JPEGImages
			|——SegmentationClass
			|——SegmentationObject
		|——test
			|——VOC2007
				|——Annotations
				|——ImageSets
				|——JPEGImages
				|——SegmentationClass
				|——SegmentationObject


## 开始训练

### 训练模型

1.进入解压后源码包根目录。

	cd /${模型文件夹名称} 
2.运行训练脚本。

该模型支持单机单卡训练和单机8卡训练。

	      
* 单机单卡训练

    启动单卡训练。

      bash ./test/train_full_1p.sh --data_path={data_path} --validation_data_path=real_validation_path #train full_1p

    测试单卡性能。

      bash ./test/train_performance_1p.sh --data_path={data_path} --validation_data_path=real_validation_path #train 1p_performance  

* 单机8卡训练

    启动8卡训练。

      bash ./test/train_full_8p.sh --data_path={data_path} --validation_data_path=real_validation_path --loss_scale=128.0 #train full_8p

    测试8卡性能 。
   
      bash ./test/train_performance_8p.sh --data_path={data_path} --validation_data_path=real_validation_path --loss_scale=128.0 #train 8p_performance
* 启动单卡/8卡评估

      bash test/train_eval.sh --data_path={data_path} --pth_path=real_pre_train_model_path #此处data_path为验证集路径
--data_path参数填写数据集路径(除单卡/8卡评估模式外，此参数均为训练集路径，单卡/8卡评估模式此参数为验证集路径)；  
--validation_data_path参数填写测试集路径；  
--loss_scale参数是混合精度loss scale大小，默认是128.0；  
--pth_path参数填写训练过程中生成的权重文件路径（默认存储在"models/"文件夹下）。

    公共参数：
    --data                              //数据集路径
    --addr                              //主机地址
    --workers                           //加载数据进程数      
    --epoch                             //重复训练次数
    --batch-size                        //训练批次大小
    --lr                                //初始学习率
    --momentum                          //动量，默认：0.9
    --weight_decay                      //权重衰减，默认：0.0005
    --amp                               //是否使用混合精度
    --loss-scale                        //混合精度lossscale大小
    --opt-level                         //混合精度类型
    多卡训练参数：
    --multiprocessing-distributed       //是否使用多卡训练
    --device-list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡

## 训练结果展示

**表 2** 训练结果展示表

| NAME	       | 	ACC@1   |  FPS  |   Epochs  |AMP_Type|
|-------------|----------|-------|-----------|--------|
| 	NPU1.8-1P  | 	0.67807 |	  346	|    240	|       O1|		
|  NPU1.8-8P	 |0.6849	  | 2657	    | 240	     | O2       |
| 	NPU1.5-1P	 | 0.67662	 | 54	      | 240	     | O1       |
| 	NPU1.5-8P	 | 0.6783	  | 1000	    | 240	     | O2       |

## 版本说明

### 变更

2022.11.30：更新内容，重新发布。

### 已知问题

无。