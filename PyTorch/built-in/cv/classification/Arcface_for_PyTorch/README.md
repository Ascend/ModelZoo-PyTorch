# ArcFace for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

Arcface-Torch可以高效快速地训练大规模人脸识别训练集。本模型支持partial fc采样策略，当训练集中的类的数量大于100万时， 使用partial fc可以获得相同的精度，而训练性能快几倍，GPU内存更小。本模型支持多机分布式训练和混合精度训练。 

- 参考实现：

  ```
  url=https://github.com/deepinsight/insightface.git
  commit_id=e78eee59caff3515a4db467976cf6293aba55035
  code_path=insightface/recognition/arcface_torch/
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=ModelZoo-PyTorch/PyTorch/built-in/cv/classification/Arcface_for_PyTorch
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

  | 配套        | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动   | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)|
  | Apex | [0.1](https://gitee.com/ascend/apex/tree/master/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖（根据模型需求，按需添加所需依赖）。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   主要参考 https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc#4-download 进行glint360k数据集准备。用户自行按照原始代码仓指导获取训练所需的数据集。
   准备好数据集后放到glint360k目录下

   ```
   ├── glint360k
         ├── agedb_30.bin（约73.0M）               
         ├── calfw.bin（约70.7M)
         ├── cfp_ff.bin（约85.2M)
         ├── cfp_fp.bin（约75.9M)
         ├── cplfw.bin（约59.7M)
         ├── lfw.bin（约61.7M)
         ├── train.idx（约342.3M)
         ├── train.rec（约129.5G)
         ├── vgg2_fp.bin（56.8M）
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理

    - 本模型不涉及

## 获取预训练模型（可选）

- 本模型不涉及

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机8卡，4机32卡训练。

   - 单机8卡训练

     启动8卡训练

     ```
     bash ./test/train_full_8p.sh   
     ```
   
     也可使用模型原生方式启动训练
   
     ```
     source ./test/env_npu.sh
     python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train.py configs/glint360k_r100.py
     ```
     
     模型训练脚本参数说明如下：
   
     ```
     公共参数：
     configs/glint360k_r100.py                            //训练配置
     ```
     
     训练完成后，权重文件保存在./work_dirs/glint360k_r100下，并输出模型训练精度和性能信息。
     
   - 4机32卡训练
   
     请参考[PyTorch模型多机多卡训练适配指南](https://gitee.com/ascend/pytorch/blob/master/docs/zh/PyTorch%E6%A8%A1%E5%9E%8B%E5%A4%9A%E6%9C%BA%E5%A4%9A%E5%8D%A1%E8%AE%AD%E7%BB%83%E9%80%82%E9%85%8D%E6%8C%87%E5%8D%97.md)中的“多机多卡训练流程”-“准备环境”章节进行环境设置，然后在每台服务器上使用如下命令启动训练
   
     ```
     source ./test/env_npu.sh
     python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=4 --node_rank=节点id --master_addr="主服务器地址" --master_port=12581 train.py configs/glint360k_r100_32gpus.py
     注：节点id各服务器分别填0-3
     ```
   

# 训练结果展示

**表 2**  训练结果展示表

| NAME     | Accuracy-Highest |  FPS | AMP_Type |
| -------  | -----  | ---: | -------: |
| 8p-竞品A  | lfw: 0.99833 cfp_fp: 0.99257 agedb_30: 0.98633 | 4163 |       O1 |
| 8p-NPU   | lfw: 0.99850 cfp_fp: 0.99229 agedb_30: 0.98533 | 4158 |       O1 |
| 32p-NPU | lfw: 0.99817 cfp_fp: 0.99214 agedb_30: 0.98683 | 14515 | O1 |

备注：一定要有竞品和NPU。

# 版本说明

## 变更

2022.08.23：首次发布

## 已知问题

**_当前发行版本中存在的问题描述。_**

1.因sklearn自身bug，若运行环境为ARM，则需要手动导入so，以下是root python环境里的示例
```export LD_PRELOAD=/usr/local/python3.7.5/lib/python3.7/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0```


2.如果遇到了这个报错“OSError: libarmpl_lp64_mp.so: cannot open shared object file: No such file or directory”，则可以参照这个issue处理，https://github.com/apache/mxnet/issues/19234 。











