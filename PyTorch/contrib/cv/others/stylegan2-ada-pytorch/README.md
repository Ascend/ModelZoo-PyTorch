# StlyeGAN2-ADA for PyTorch

- [StlyeGAN2-ADA for PyTorch](#stlyegan2-ada-for-pytorch)
- [概述](#概述)
  - [简述](#简述)
- [准备训练环境](#准备训练环境)
  - [准备环境](#准备环境)
  - [准备数据集](#准备数据集)
- [开始训练](#开始训练)
  - [训练模型](#训练模型)
- [训练结果展示](#训练结果展示)
- [生成图片](#生成图片)
- [版本说明](#版本说明)
  - [变更](#变更)
  - [已知问题](#已知问题)



# 概述

## 简述

StlyeGAN2-ADA提出了一种自适应鉴别器增强机制，在有限的数据体系中显著稳定训练，防止过拟合导致训练发散。

StlyeGAN2-ADA-Pytorch使用FID(Fréchet inception distance)作为评价模型训练效果的指标,FID值越小模型效果越好。

- 参考实现：

  ```
  url=https://github.com/NVlabs/stylegan2-ada-pytorch
  commit_id=6f160b3d22b8b178ebe533a50d4d5e63aedba21d
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/others/
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
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```

- 下载inception模型用于计算FID。
  - 链接：https://pan.baidu.com/s/1CBiKXaBzS8A0IGxcBivbTA 提取码：ince
  - 进入模型代码目录下，将inception_weight.pth文件保存到 `./inception/` 文件夹下。


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可下载[AFHQ数据集](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq)或选择其他数据集。

   以AFHQ数据集为例，数据集目录结构参考如下所示。

   ```
    afhqwild
    ├── 00000
    │   ├── img00000000.png
    │   ├── img00000001.png
    │   ├── img00000002.png
    │   ├── ...
    ├── 00001
    │   ├── img00001000.png
    │   ├── img00001001.png
    │   ├── img00001002.png
    │   ├── ...
    ├── 00002
    │   ├── img00002000.png
    │   ├── img00002001.png
    │   ├── img00002002.png
    │   ├── ...
    ├── 00003
    │   ├── img00003000.png
    │   ├── img00003001.png
    │   ├── img00003002.png
    │   ├── ...
    ├── 00004
    │   ├── img00004000.png
    │   ├── img00004001.png
    │   ├── img00004002.png
    │   ├── ...
    └── dataset.json            
   ```
    > **说明：** 
    >该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理。

   可以使用`dataset_tool.py`对原始数据集进行处理：
   ```bash
   python dataset_tool.py --source=~/downloads/afhq/train/wild --dest=~/data/afhqwild_64.zip \
       --width=64 --height=64
   ```
   得到的zip文件可以直接用于训练，也可解压后用于训练。可通过`--width`与`--height`指定图片分辨率，使用大分辨率时按需调整batch_size。


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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/    
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/   
     ```

   --data\_path参数填写数据集路径，可以传入zip文件的路径如~/data/afhqwild_64.zip，或解压zip文件后传入解压后文件夹的路径如~/data/afhqwild_64。
   
   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data                              //数据集路径
   --outdir                            //输出路径
   --kimg                              //重复训练图片数
   --batch                             //训练批次大小
   --fp32                              //是否使用fp32训练
   --snap                              //保存模型间隔
   多卡训练参数：
   --gpus                              //训练使用卡数
   ```

   训练完成后，权重文件保存在当前路径下的`out/<ID>-mydataset-<Configurations>`文件夹下，并输出模型训练精度和性能信息，其中ID由程序生成，Configurations与训练参数有关。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | FID   | FPS | Kimgs   |
| ------- | ----- | ---: | ------: |
| 1p-竞品 | -     |  44.82| 1 |
| 1p-NPU  | -     |  13.09 | 1 |
| 8p-竞品 | 6.70  | 318.47 | 3000 |
| 8p-NPU  | 6.63 | 67.38 | 3000 |


# 生成图片

可以使用`generate.py`，传入以.pkl文件保存的训练好的模型生成图片：

```bash
python generate.py --outdir=output_path --seeds=1-10 --network=real_pre_train_model_path
```

--seeds为生成图片的种子，--outdir指定图片保存路径，--network为.pkl文件的路径。

# 版本说明

## 变更

2022.10.13：首次发布。

## 已知问题

由于NPU硬件和一些算子的限制，输入图片的尺寸必须不大于64×64。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md