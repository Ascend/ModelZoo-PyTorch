# EfficientNet-B3 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

EfficientNet是Google研究小组2019年在论文《EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks》中推出的新模型(Tan and Le 2019)，该模型基于网络深度、宽度和输入分辨率三个维度的缩放来寻找最优模型。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/pycls
  commit_id=ee89cecb0e295b8037843e7e28344b156a847554
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url} # 克隆仓库的代码
  cd {code_path} # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
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
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。
  ```
  pip install pycls
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集imagenet，将数据集上传到服务器并解压。

   注：为方便运行，数据集应该存放在如下文件路径中：
   ```
   ./Efficientnet-B3/pycls/datasets/data
   # 数据集软链接方式
   ln -s {data/path} EfficientNet-B3/pycls/datasets/data
   ```
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
   >该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理。
- 在pycls/datasets/loader.py中修改数据集的路径，你可以将变量_DATA_DIR修改为你的imagenet数据集的路径。


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
     bash ./test/train_performance_1p.sh --data_path={data/path}  # 1p性能
     bash ./test/train_full_1p.sh --data_path={data/path}         # 1p精度 
     ```

   - 单机8卡训练

     启动8卡训练。
     ```
     bash ./test/train_performance_8p.sh --data_path={data/path}   # 8p性能
     bash ./test/train_full_8p.sh --data_path={data/path}          # 8p精度 完成100个epoch训练大约27h
     bash ./test/train_eval_8p.sh --data_path={data/path}    # 8p验证 
     ```

   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --device                            //使用设备，gpu或npu
   --cfg                               //使用yaml配置文件路径
   --rank_id                           //默认卡号
   --device_id                         //默认设备号
   ```
   

# 训练结果展示

**表 2**  训练结果展示表

| Acc@1  | FPS  | Npu_nums | Epochs | AMP_Type | Torch |
| :----: | :--: | :------: | :----: | :------: | :---: |
|   -    | 267   |    1     |  1   |    O2    |  1.5  | 
| 77.3418 | 1558  |    8     |  100   |    O2    |  1.5  |
|   -    | 356   |    1     |  1   |    O2    |  1.8  |
| 77.0613 | 2236  |    8     |  100   |    O2    |  1.8  | 


# 版本说明
2022.08.01：更新pytorch1.8版本，重新发布。

## 已知问题
无。






