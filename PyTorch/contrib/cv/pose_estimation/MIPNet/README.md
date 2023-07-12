# MIPNet for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述
MIPNet打破了传统2D人类姿态预测中自顶而下算法里一个边界框中只含有一个实体/人的关键假设，允许一个边界框有多个实例，能够恢复多个被遮挡者。同样是通过热点图预测，与HRNet只关注了前景的人相比，MIPNet允许通过变换λ因子，对边框内多个实例的预测。

- 参考实现：

  ```
  url=https://github.com/rawalkhirodkar/MIPNet.git
  commit_id=505c92ec59ac79686a217dac45eb188fc38b8499
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/pose_estimation
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
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```

- 安装crowdpose。

  下载并切换到文件夹进行安装。
  ```
  git clone https://github.com/Jeff-sjtu/CrowdPose.git
  cd CrowdPose/crowdpose-api/PythonAPI/

  make install
  python setup.py install --user
  ```
  
  校验输入，不报错即成功。
  ```
  import crowdposetools
  ```

## 准备数据集

1. 获取数据集。

  - 下载 COCO 数据集，将数据集上传到服务器任意路径下并解压。

    解压后，数据集目录结构如下所示：

    ```
    ├── COCO
    │   │   ├── annotations
    |   |   │   │   ├── instances_val2017.json
    |   |   │   │   ├── instances_train2017.json  
    |   |   │   │   ├── captions_train2017.json
    |   |   │   │   ├── ……
    │   │   ├── images
    |   |   │   │   ├──train2017
    |   |   |   |   │   │   ├──xxxx.jpg
    |   |   │   │   ├──val2017
    |   |   |   |   │   │   ├──xxxx.jpg    
    │   │   ├── labels
    |   |   │   │   ├──train2017
    |   |   |   |   │   │   ├──xxxx.txt
    |   |   │   │   ├──val2017
    |   |   |   |   │   │   ├──xxxx.txt
    |   |   ├──test-dev2017.txt  
    |   |   ├──test-dev2017.shapes
    |   |   ├──train2017.txt
    |   |   ├──……
    ```
## 获取预训练模型
从百度网盘链接：https://pan.baidu.com/s/1hw6EmwYdQDF_yYyc7uNolw 获取（提取码：mip5），或者直接从github原readme中下载。下载后放置于源码根目录 “./lib/models/imagenet/”下。

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
     # training 1p performance
     bash ./test/train_performance_1p.sh --data_path=xxx 
     # --data_path= 实际数据集下载路径,下同


     # training 1p finetune
     bash ./test/train_finetune_1p.sh --data_path=xxx --pth_path=xxx
      # --pth_path= 导入的checkpoint路径
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     # training 8p accuracy
     bash ./test/train_full_8p.sh --data_path=xxx 

     # training 8p performance
     bash ./test/train_performance_8p.sh --data_path=xxx 

     ```

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --cfg                               //实验参数文件
   --locak_rank                        //多卡训练指定训练用卡
   --is_distributed                    //8卡训练：1 / 单卡训练 0    
   --perf                              //performance模式 1/ full模式 0
   --check_point                       //可导入pth文件路径

   ```

   日志和权重文件保存在如下路径。

   ```
   ./test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log       # training detail log(include result)
   ./output/coco/pose_hrnet/w48_384x288_adam_lr1e-3/                 # checkpoits

   ```

# 训练结果展示

**表 2**  训练结果展示表

| 名称    |  FPS   |  Ap |
| :------: | :------: | :------: | 
| 1p-GPU | 39.0081  | ----- |
| 1p-NPU  | 40.55 | -----| 
| 8p-GPU | 212.988 | 78.2 | 
| 8p-NPU  | 248.914 | 78.0 | 

# 版本说明

## 变更

2022.10.17：首次发布。

## 已知问题

无。


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md








