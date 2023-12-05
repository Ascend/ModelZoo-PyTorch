# SCRFD for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

SCRFD是通过NAS（神经网络搜索）得到的一个目标检测模型，通过使用ResNet作为主干网络，融入PAFPN、ATSS Assigner等模块，可以获得更好的精度、更高的性能。


- 参考实现：

  ```
  url=https://github.com/deepinsight/insightface.git
  commit_id=babb9a58bbc42ae4b648acdbb803159a35f53db3
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/detection
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  ```shell
  pip install -r requirements.txt
  ```

- 安装mmcv（如果环境中有mmcv，请先卸载再执行以下步骤）。
  ```shell
  git clone -b v1.4.8 https://github.com/open-mmlab/mmcv.git
  bash tools/build_mmcv.sh
  ```

- 安装mmdet（如果环境中有mmdet，请先卸载再执行以下步骤）。
  ```shell
  pip3.7 install -v -e .
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括WIDER_FACE，将数据集上传到服务器任意路径下并解压。

   数据集目录结构参考如下所示。

   ```
   ├── WIDERFace
         ├──train
              ├──images
                    │──图片1
                    │──图片2
                    │   ...       
              ├──labelv2.txt              
         ├──val  
              ├──images
                    │──图片1
                    │──图片2
                    │   ...       
              ├──gt  
                    │──*.gt    
              ├──labelv2.txt 
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

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。
     
     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```
   

--data_path参数填写数据集路径，需写到数据集的一级目录。

模型训练脚本参数说明如下。

```
公共参数：
--work-dir                          //日志和模型保存目录
--no-validate                       //是否禁用训练中的eval流程
--perf                              //是否进行性能测试
--seed                              //随机数种子设置
--local_rank                        //当前进程的rank号
--master_addr                       //主进程IP地址
--master_port                       //主进程端口号
```

训练完成后，权重文件保存在工作目录下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 |  FPS | Epochs | AMP_Type | Torch_Version |
| :-----: | :---: | :--: | :----: | :------: | :------: |
| 1p-竞品V | -     |  78 | 3      |        - |     1.5 |
| 8p-竞品V | [95.16, 93.87, 83.05] | 278 | 640    |        - |     1.5 |
| 1p-NPU  | -     |  60 | 3      |       O2 |    1.8 |
| 8p-NPU  | [94.58, 93.38, 81.52] | 170 | 640    |       O2 |    1.8 |

# 版本说明

## 变更

2023.02.22：更新readme，重新发布。

2022.09.20：首次发布。

## FAQ


目前可能会出现mmpycocotools与pycocotools两个第三方库冲突的问题，如果出现pycocotools相关问题，需要首先将mmpycocotools和pycocotools全部卸载，然后重装mmpycocotools即可解决冲突问题。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
