# FixRes for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

  FixRes是一个图像分类网络，该模型使用较低分辨率图像输入对ResNet50网络进行训练，并使用较高分辨率图像输入对训练好的模型进行finetune，最终使用较高分辨率进行测试，以此解决预处理过程中图像增强方法不同引入的偏差。
- 参考实现：
    ```
    url=https://github.com/facebookresearch/FixRes
    branch=master
    commit_id=c9be6acc7a6b32f896e62c28a97c20c2348327d3
    ```

- 适配昇腾 AI 处理器的实现：

    ```
    url=https://gitee.com/ascend/ModelZoo-PyTorch
    code_path=PyTorch/contrib/cv/classification
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
  | 硬件       | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```
  pip3.7 install -r requirements.txt
  ```

## 准备数据集


1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，将数据集上传到服务器任意路径下并解压。

   以ImageNet数据集为例，数据集目录结构参考如下所示。

   ```
   ├── ImageNet2012
         ├──train
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...   
              ├──...                     
         ├──val  
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                 │──图片1
                    │──图片2
                    │   ...                
   ```


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。在训练之前需要在源码包根目录下创建“train_cache”文件夹，用来存放训练时产生的文件。

   - 单机单卡训练
    
     ```
     bash ./test/train_performance_1p.sh --data_path=数据集路径 
     bash ./test/train_full_1p.sh --data_path=数据集路径
     bash ./test/finetune_full_1p.sh --data_path=数据集路径 --pth_path=train_model.pth    
     ```

   - 单机8卡训练

     ```
     bash ./test/train_performance_8p.sh --data_path=数据集路径     
     bash ./test/train_full_8p.sh --data_path=数据集路径
     bash ./test/finetune_full_8p.sh --data_path=数据集路径 --pth_path=train_model.pth 
     ```

3. 模型训练脚本参数说明如下。

    ```
    公共参数：
    --data_path                         //数据集路径
    --addr                              //主机地址
    --imnet_path                        //数据集路径
    --num_tasks                         //使用卡数 
    --epochs                            //重复训练次数
    --batch                             //训练批次大小
    --learning_rate                     //初始学习率
    --local_rank                        //训练指定用卡
    --pth_path                          //pth文件存放路径  
    ```

# 训练结果展示

**表 2**  训练结果展示表

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 590 (train) / 711(fineutune) | 1        | 120 + 60 | O1       |
| 72.9% (72.1% before finetune)    | 3318(train) / 3515(finetune) | 8        | 120 + 60 | O1       |

# 版本说明

## 变更

2023.1.30：更新readme，重新发布。


## 已知问题

暂无。