# SSD-MobilenetV2 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述
SSD网络是继YOLO之后的one-stage目标检测网络，是为了改善YOLO网络设置的anchor设计的太过于粗糙而提出的，其设计思想主要是应用多尺度多长宽比的密集锚点设计和特征金字塔。在本模型中用Mobilenet-V2代替其原始模型中的VGG网络。

MobileNet-V2网络是由google团队在2018年提出的，相比MobileNet-V1网络，准确率更高，模型更小。 该网络中的主要亮点 ：
Inverted Residuals （倒残差结构 ）；Linear Bottlenecks（结构的最后一层采用线性层）。

- 参考实现：

  ```
    url=https://github.com/qfgaohao/pytorch-ssd.git 
    commit_id=6d7f986c4ac07744bf8375d1d01f9493663c47df
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

    **表 1** 版本配套表

       | 配套      | 版本                                                                           |
       |------------------------------------------------------------------------------| ------------------------------------------------------------ |
       | 硬件      | [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)  |
       | NPU固件与驱动  | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)  |
       | CANN    | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
       | PyTorch | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)                       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，将数据集上传到服务器任意路径下并解压。
    ```
   wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
   wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
   wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
   ```
2. 将3个数据集放至在任意目录下新建的“VOC0712/”文件夹下。
   
    数据集目录结构如下所示：
   ```
       VOC0712
    |
    |———————— VOC2007_trainval
    |         |——————Annotations
    |         |——————ImageSets
    |         |——————JPEGImages
    |         |——————SegmentationClass
    |         |——————SegmentationObject
    |———————— VOC2012_trainval
    |         |——————Annotations
    |         |——————ImageSets
    |         |——————JPEGImages
    |         |——————SegmentationClass
    |         |——————SegmentationObject
    |———————— VOC2007_test
              |——————Annotations
              |——————ImageSets
              |——————JPEGImages
              |——————SegmentationClass
              |——————SegmentationObject
   ```




## 获取预训练模型
下载预训练模型到“models/”目录下（进入源码包根目录执行以下命令）。

  ```
wget -P models https://storage.googleapis.com/models-hao/mb2-imagenet-71_8.pth
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
     bash ./test/train_full_1p.sh --data_path=xxx
     ```
     测试单卡性能。
     ```
     bash ./test/train_performance_1p.sh --data_path=xxx
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=xxx  
     ```
      测试8卡性能。
     ```
     bash ./test/train_performance_8p.sh --data_path=xxx
     ```

   --data_path参数填写数据集路径。

3. 模型训练脚本参数说明如下。

   ```
   --data_path             数据集路径
   --dataset_type          数据集种类
   --base_net              预训练模型存放路径
   --lr                    学习率
   --num_epochs            训练epoch
   --validation_epochs     验证epoch
   --checkpoint_folder     模型保存路径
   --eval_dir              模型验证时产生文件的存放路径
   --device                使用的设备，npu或gpu
   --gpu                   设备卡号，单卡时使用
   --device_list           默认为 '0,1,2,3,4,5,6,7'，多卡时使用
   
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示



**表 2** 训练结果展示表
    
| NAME      | Acc@1 |    FPS | Epochs | AMP_Type |
|-----------|-------|-------:|--------|---------:|
| NPU1.5-1P | -     |     35 | 5      |       O2 |
| NPU1.5-8P | 0.676 |    177 | 200    |       O2 |
| NPU1.8-1P | -     |  65.69 | 20     |       O2 |
| NPU1.8-8P | 0.68  | 214.74 | 200    |       O2 |



# 版本说明

## 变更

2022.8.28：更新内容，重新发布。



## 已知问题



无。

