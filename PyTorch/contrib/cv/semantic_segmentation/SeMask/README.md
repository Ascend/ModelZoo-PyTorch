# SeMask

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述
SeMask是一个图像语义分割框架，通过以下两种技术将语义信息整合到通用的分层视觉转换器架构（例如Swin Transformer）中。首先在Transformer层之后增加了一个Semantic层，其次使用了两个解码器：一个仅用于训练的轻量级语义解码器和一个特征解码器。SeMask在微调的同时将图像的语义信息合并到预训练的基于分层Transformer的主干中，获得了更优的性能。

- 参考实现：

  ```
  url=https://github.com/Picsart-AI-Research/SeMask-Segmentation/tree/main/SeMask-FPN
  commit_id=1f599a1c5ee3c9197bd857f95e939863d3ead112
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/semantic_segmentation
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
  | 硬件 | [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | NPU固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  pip install docutils myst-parser sphinx sphinx_copybutton sphinx_markdown_tables
  pip install -e git+https://github.com/gaotongxiao/pytorch_sphinx_theme.git#egg=pytorch_sphinx_theme
  pip install cityscapesscripts
  pip install matplotlib mmcls numpy packaging prettytable
  pip install codecov flake8 interrogate pytest xdoctest yapf
  apt-get install numactl
  
  # 卸载已安装的mmcv
  pip3 uninstall mmcv
  pip3 uninstall mmcv-full
  
  # 安装新的mmcv
  pip3 install mmcv-full -f http://download.openmmlab.com/mmcv/dist/npu/torch1.8.0/index.html
  ```

## 准备数据集

1. 用户自行准备好数据集，包含训练集和验证集两部分，可选用的数据集包括Cityscapes，ADE20K等。以Cityscapes为例，可进入Cityscapes官网进行下载：https://www.cityscapes-dataset.com/。

	
2. Cityscapes目录结构如下：
	```
    Cityscapes
    ├──gtFine
    │   ├──train
    │   │   ├──aachen
    │   │   ├──......
    │   ├──val
    │   │   ├──frankfurt
    │   │   ├──......
    ├──leftImg8bit 
    │   ├──train
    │   │   ├──aachen
    │   │   ├──......
    │   ├──val
    │   │   ├──frankfurt
    │   │   ├──......
    ├──swin_small_patch4_window7_224.pth
	```

说明：该数据集的训练过程脚本只作为一种参考示例。

## 获取预训练模型

- 在`$data_path`目录下添加文件[swin_small_patch4_window7_224.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth)

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机8卡，单机单卡。

   - 单机单卡训练

     启动单卡训练

     ```
     bash ./test/train_full_1p.sh --data_path=$data_path
     ```
     ```
     bash ./test/train_performance_1p.sh --data_path=$data_path
     ```
    
     训练完成后，输出模型训练精度和性能信息。

   - 单机8卡训练

     启动8卡训练

     ```
     bash ./test/train_full_8p.sh --data_path=$data_path
     ```
     ```
     bash ./test/train_performance_8p.sh --data_path=$data_path
     ```
     `--data_path`参数填写数据集根目录。

   - 模型训练脚本参数说明如下。

      ```
      公共参数：
      --train_epochs                      //训练的总epochs数
      --workers                           //dataloader开启的线程数
      ```
    
     训练完成后，权重文件默认会写入到和test文件同一目录下，并输出模型训练精度和性能信息到网络脚本test下output文件夹内。

# 训练结果展示


  **表 2**  训练结果展示表

  | NAME   | Acc@1 | FPS   | PyTorch_version |
  |--------|-------|-------|-----------------|
  | GPU-1P | -  | 9.521  | 1.5             |
  | GPU-8P | 79.02  | 63.15   | 1.5             |
  | NPU-1P | -  | 8.153  | 1.5             |
  | NPU-8P | 79.02  | 62.126 | 1.5             |



# 版本说明

## 变更

2022.12.21：首次发布。


## 已知问题
无。

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md 