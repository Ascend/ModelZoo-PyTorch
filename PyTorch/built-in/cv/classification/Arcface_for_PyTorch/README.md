# ArcFace for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

Arcface-Torch可以高效快速地训练大规模目标识别训练集。本模型支持partial fc采样策略，当训练集中的类的数量大于100万时， 使用partial fc可以获得相同的精度，而训练性能快几倍，GPU内存更小。本模型支持多机分布式训练和混合精度训练。 

- 参考实现：

  ```
  url=https://github.com/deepinsight/insightface.git
  commit_id=e78eee59caff3515a4db467976cf6293aba55035
  ```
  
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |
  | PyTorch 1.11   | - |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行下载数据集，可参照原始代码仓指导获取训练所需的数据集。
   将数据集放到源码包根目录下新建的glint360k目录中，数据集目录结构参考如下所示。

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


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡，单机8卡，4机32卡训练。

   - 单机单卡训练

     启动单卡训练

     ```
     bash ./test/train_performance_1p.sh  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练

     ```
     bash ./test/train_full_8p.sh  # 8卡精度
     
     bash ./test/train_performance_8p.sh  # 8卡性能
     ```

   - 4机32卡训练

     请参考[PyTorch模型多机多卡训练适配指南](https://gitee.com/ascend/pytorch/blob/v1.8.1-3.0.rc2/docs/zh/PyTorch%E6%A8%A1%E5%9E%8B%E5%A4%9A%E6%9C%BA%E5%A4%9A%E5%8D%A1%E8%AE%AD%E7%BB%83%E9%80%82%E9%85%8D%E6%8C%87%E5%8D%97.md)中的“多机多卡训练流程”-“准备环境”章节进行环境设置，然后在每台服务器上使用如下命令启动训练。

     ```
     source ./test/env_npu.sh
     python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=4 --node_rank=节点id --master_addr="主服务器地址" --master_port=12581 train.py configs/glint360k_r100_32gpus.py
     注：节点id各服务器分别填"0-3"
     ```
   
   模型训练脚本参数说明如下：
   
   ```
   公共参数：
   --perf_steps                                         //设置性能训练的迭代数
   --profiling                                          //是否开启profiling，默认关闭
   --start_step	                                     //开始的迭代数
   --stop_step                                          //停止的迭代数
   --local_rank                                         //训练rank号
   ```
   
   训练完成后，权重文件保存在./work_dirs/glint360k_r100下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

| NAME     | Accuracy-Highest |  FPS | Epochs | AMP_Type | Torch_Version |
| :-----:  | :---:  | :--: | :------: | :------: | :------: |
| 8p-竞品A  | lfw: 0.99833 cfp_fp: 0.99257 agedb_30: 0.98633 | 4163 | 20 |       O1 |    1.5 |
| 1p-NPU | - | 347.909 | 1 | O1 | 1.8 |
| 8p-NPU   | lfw: 0.99850 cfp_fp: 0.99314 agedb_30: 0.98600 | 4148.302 | 20 |       O1 |    1.8 |
| 32p-NPU | lfw: 0.99817 cfp_fp: 0.99214 agedb_30: 0.98683 | 14515 | 20 | O1 | 1.8 |


# 版本说明

## 变更

2023.02.20：更新readme，重新发布。

2022.08.23：首次发布。

## FAQ

1. 因sklearn自身bug，若运行环境为ARM，则需要手动导入so，以下是root python环境里的示例。

   ```
   export LD_PRELOAD=/usr/local/python3.7.5/lib/python3.7/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
   ```

2. 若在ARM环境中，使用三方库**mxnet**遇到这个报错“OSError: libarmpl_lp64_mp.so: cannot open shared object file: No such file or directory”，可参考以下方法对**mxnet**进行源码安装。

   ```
   pip uninstall mxnet --y
   apt update; apt install libopencv-dev
   wget https://archive.apache.org/dist/incubator/mxnet/1.9.1/apache-mxnet-src-1.9.1-incubating.tar.gz 
   tar -xvf apache-mxnet-src-1.9.1-incubating.tar.gz; cd apache-mxnet-src-1.9.1-incubating
   cp config/linux_arm.cmake config.cmake
   mkdir build; cd build
   cmake ..
   cmake --build . -j64
   cd ../
   python -m pip install --user -e ./python
   ```