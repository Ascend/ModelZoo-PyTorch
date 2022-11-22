# EfficientNet-B5 for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

EfficientNet-B5网络模型属于EfficientNet系列网络，是由谷歌团队提出的一种图像分类网络。该系列网络的基础网络EfficientNet-B0通过神经网络搜索（NAS）搜索得出，随后通过复合缩放策略对EfficientNet-B0进行分辨率、深度和宽度三个维度上的同时缩放，得到了EfficientNet B1-B7，实现了网络在效率和准确率上的优化。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/pycls.git
  commit_id=0ddcc2b25607c7144fd6c169d725033b81477223
  ```

- 适配昇腾 AI 处理器的实现：

    ```
    url=https://gitee.com/ascend/ModelZoo-PyTorch.git
    code_path=PyTorch/contrib/cv/classification
    ```

- 通过Git获取代码方法如下：
  
    ```
    git clone {url}        # 克隆仓库的代码   
    cd {code_path}         # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
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
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

    用户自行获取原始数据集，使用的开源数据集为ImageNet，下载地址为[http://www.image-net.org/](http://www.image-net.org/)，将数据集上传到服务器任意路径下并解压。
       数据集目录结构如下所示：  

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
   数据集软链接方式:
     ```
       ./Efficientnet-B5/pycls/datasets/data
       # 数据集软链接方式
       ln -s /{data/path}/imagenet /{data/path}/pycls/pycls/datasets/data/imagenet
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
     bash ./test/train_full_1p.sh --data_path=xxx
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=xxx
     ```

   --data\_path参数填写数据集根目录

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --cfg                               //使用yaml配置文件路径
   --rank_id                           //默认卡号
   --device_id                         //默认设备号
   ```
   训练完成后，权重文件默认会写入到和test文件同一目录下，并输出模型训练精度和性能信息到网络脚本test下output文件夹内。
# 训练结果展示

**表 2**  训练结果展示表

| Acc@1  | FPS  | Npu_nums | Epochs | AMP_Type | Torch |
| :----: | :--: | :------: | :----: | :------: | ----- |
|   -    |  47  |    1     |  100   |    O2    | 1.5   |
| 78.595 | 384  |    8     |  100   |    O2    | 1.5   |
|   -    |  55  |    1     |  100   |    O2    | 1.8   |
| 79.092 | 430  |    8     |  100   |    O2    | 1.8   | 


# 版本说明
## 变更
2022.08.01：更新pytorch1.8版本，重新发布。

2020.12.23：首次发布。
## 已知问题
无。