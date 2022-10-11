# YOLOR

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)



# 概述

## 简述

  YOLOR提出了一个统一的网络来同时编码显式知识和隐式知识，在网络中执行了kernel space alignment（核空间对齐）、prediction refinement（预测细化）和 multi-task learning（多任务学习），同时对多个任务形成统一的表示，基于此进行目标识别。

- 参考实现：

  ```
  url=https://github.com/WongKinYiu/yolor
  commit_id=b168a4dd0fe22068bb6f43724e22013705413afb
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

   | 配套  | 版本  |
   |---|---|
   | 固件与驱动  | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)   |
   | CANN  |  [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2)  |
   | PyTorch  | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)  |




- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 进入源码包根目录，执行以下命令，下载coco数据集。数据集信息包含图片、labels图片以及annotations：
	```
	cd /${模型文件夹名称}
	bash scripts/get_coco.sh
	```
2. coco目录结构如下：
	```
   coco
   |-- LICENSE
   |-- README.txt
   |-- annotations
   |   |-- instances_val2017.json
   |-- images
   |   |-- test2017
   |   |-- train2017
   |   |-- val2017
   |-- labels
   |   |-- train2017
   |   |-- train2017.cache3
   |   |-- val2017
   |   |-- val2017.cache3
   |-- test-dev2017.txt
   |-- train2017.cache
   |-- train2017.txt
   |-- val2017.cache
   |-- val2017.txt
	```
说明：该数据集的训练过程脚本只作为一种参考示例。


# 开始训练

## 训练模型
1. 进入解压后的源码包根目录。

    ```
    cd /${模型文件夹名称} 
    ```

2. 运行训练脚本。

   该模型支持单机8卡训练。

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/coco  
     ```

   模型训练脚本参数说明如下。

    ```
    公共参数：
    --data_dir                              //数据集路径      
    --epoch                             //重复训练次数
    --batch_size                        //训练批次大小
    --lr                                //初始学习率，默认：0.01
    --amp_cfg                               //是否使用混合精度
    --loss_scale_value                        //混合精度lossscale大小
    --opt_level                         //混合精度类型
    多卡训练参数：
    --multiprocessing_distributed       //是否使用多卡训练
    --device_list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
    ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度信息。

  

3. 运行性能脚本。  
                                                                                                                                                                
   该模型支持单机单卡性能训练和单机8卡性能训练。                                                                                                                                    

   - 单机单卡性能训练                                                                                                                                                                                 
    
     启动单卡性能训练。 

     ```
     bash ./test/train_performance_1p.sh --data_path=/data/coco  
     ```                                                                                                                                                                                              

   - 单机8卡性能训练                                                                                                                                                                                 
    
     启动8卡性能训练。

     ```
     bash ./test/train_performance_8p.sh --data_path=/data/coco  
     ```

     --data\_path参数填写数据集路径。

    模型性能脚本参数说明如下。

    ```
    公共参数：
    --data_dir                              //数据集路径      
    --epoch                             //重复训练次数
    --batch_size                        //训练批次大小
    --lr                                //初始学习率，默认：0.01
    --amp_cfg                               //是否使用混合精度
    --loss_scale_value                        //混合精度lossscale大小
    --opt_level                         //混合精度类型
    多卡训练参数：
    --multiprocessing_distributed       //是否使用多卡训练
    --device_list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
    ```
   
    训练完成后，权重文件保存在当前路径下，并输出模型性能信息。

# 训练结果展示


  **表 2**  训练结果展示表

  | NAME   | Acc@1 | FPS   | PyTorch_version |
  |--------|-------|-------|-----------------|
  | NPU-1P | 51.6  | 14.7  | 1.5             |
  | NPU-8P | 51.6  | 111   | 1.5             |
  | NPU-1P | 51.6  | 18.9  | 1.8             |
  | NPU-8P | 51.6  | 143.3 | 1.8             |


# 版本说明

## 变更

2022.09.22：更新内容，重新发布。

2021.07.23：首次发布

## 已知问题
无。