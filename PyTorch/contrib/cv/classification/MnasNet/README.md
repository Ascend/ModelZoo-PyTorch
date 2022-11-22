# MnasNet for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

MnasNet是Google研究小组2019年在论文《MnasNet: Platform-Aware Neural Architecture Search for Mobile》中推出的新模型。Mnasnet网络是介于mobilenetV2和mobilenetV3之间的一个网络，这个网络是采用强化学习搜索出来的一个网络，是谷歌提出的一个轻量化网络。

- 参考实现：

  ```
  url=https://github.com/pytorch/vision/blob/master/torchvision/models/mnasnet.py
  commit_id=91e03b91fd9bab19b4c295692455a1883831a932
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
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
  |  硬件      |  [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)  |
  | NPU固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN版本      | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install torchvision==0.9.1
  pip install decorator
  pip install sympy
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集imagenet，将数据集上传到服务器并解压。

   数据集目录结构参考如下所示:

   ```
    ├── ImageNet
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

   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。

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
      bash ./test/train_full_1p.sh --data_path=real_data_path         # 1p精度
      bash ./test/train_performance_1p.sh --data_path=real_data_path  # 1p性能
      ```
   
   - 单机8卡训练
   
     启动8卡训练。
   
     ```
      bash ./test/train_full_8p.sh --data_path=real_data_path            # 8p精度 完成300个epoch训练大约16.5h
      bash ./test/train_performance_8p.sh --data_path=real_data_path   # 8p性能
      bash ./test/train_eval_8p.sh --data_path=real_data_path   		# 8p验证
     ```
   
   --data\_path参数填写数据集路径。
   
   模型训练脚本参数说明如下。
   
   ```
    公共参数：
    --device                            //使用设备，gpu或npu
    --workers                           //加载数据进程数      
    --epochs                             //重复训练次数
    --batch-size                        //训练批次大小
    --lr                                //初始学习率，默认：0.1
    --momentum                          //动量，默认：0.9
    --weight-decay                      //权重衰减，默认：0.0001
    多卡训练参数：
    --multiprocessing-distributed       //是否使用多卡训练
    --device-list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| Acc@1  | FPS  | Npu_nums | Epochs | AMP_Type | Torch |
| :----: | :--: | :------: | :----: | :------: | :---: |
|   -    | 173  |    1     |   1    |    O2    |  1.5  |
| 73.045 | 9188 |    8     |  300   |    O1    |  1.5  |
|   -    | 2569.8 |    1     |   1    |    O1    |  1.8  |
| 72.819 | 14413.78 |    8     |  300   |    O1    |  1.8  |

# 版本说明

## 变更

2022.08.01：更新pytorch1.8版本，重新发布。

## 已知问题

无。











