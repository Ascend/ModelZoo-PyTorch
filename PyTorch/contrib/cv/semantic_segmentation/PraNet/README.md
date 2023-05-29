# Pranet for PyTorch
-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述
PraNet是一个新的网络结构，用于从结肠镜图像中自动分割息肉。实验表明，在数据集中，PraNet模型始终比其它对比模型表现出更大的优势。PraNet具有较强的学习能力、泛化能力和实时分割效率。

- 参考实现：

  ```
  url=https://github.com/DengPingFan/PraNet
  commit_id=59206b0591261bc7f6ef0e3c83efd5e30a357d7
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
  | 固件与驱动 | [1.0.15](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC1](https://www.hiascend.com/software/cann/commercial?version=5.1.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖（根据模型需求，按需添加所需依赖）。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   Kvasir数据集包括开放访问的胃肠道息肉图像数据集和相应的分割掩码，由经验丰富的胃肠病学家手动注释和验证。以将数据集放置到/data目录下为例，训练集和验证集图片分别位于“./data/TrainDataset”和“/data/TestDataset/”文件夹路径下，由于验证集使用Kvasir数据集，因此验证集的路径为“./data/TestDataset/Kvasir”，数据集目录结构参考如下所示。

   ```
    data/
    ├── TestDataset
    │   └── Kvasir
    │       ├── images
    │       └── masks
    └── TrainDataset
        ├── images
        └── masks
   ```
    构建软链接
    ```
    mkdir data
    cd data
    ln -s train_path/TestDataset
    ```

   > **说明：** 
   >数据集路径'train_path'以用户自行定义的路径为准

2. 数据预处理（按需处理所需要的数据集）。

## 获取预训练模型

1. 创建文件夹。
    ```
    snapshots/PraNet_Res2Net/
    ```
2. 下载预训练权重，将权重移动至创建好的路径下。
    ```
    snapshots/PraNet_Res2Net/PraNet-19.pth
    ```
    PraNet-19.pth 下载地址为[download link (Google Drive)](https://drive.google.com/file/d/1pUE99SUQHTLxS9rabLGe_XTDwfS6wXEw/view?usp=sharing)。

    Res2Net 权重 下载地址为[download link (Google Drive)](https://drive.google.com/file/d/1_1N-cx1UpRQo7Ybsjno1PAg4KE1T9e5J/view?usp=sharing)放置在模型根目录文件夹下。

# 开始训练

## 训练模型
1. 进入解压后的源码包根目录。

    ```
    cd /${模型文件夹名称} 
    ```

2. 运行训练脚本。

    ```bash
    # training 1p accuracy
    bash ./test/train_full_1p.sh --train_path=./data/TrainDataset

    # training 1p performance
    bash ./test/train_performance_1p.sh --train_path=./data/TrainDataset

    # training 8p accuracy
    bash ./test/train_full_8p.sh --train_path=./data/TrainDataset

    # training 8p performance
    bash ./test/train_performance_8p.sh --train_path=./data/TrainDataset

    # finetuning
    bash test/train_finetune_1p.sh --train_path=./data/TrainDataset

    # online inference demo 
    python3 demo.py
    ```
    --train_path以用户数据集实际存放路径为准。

日志路径:
    
    test/output/devie_id/train_${device_id}.log           # training detail log
    test/output/devie_id/PraNet_bs16_8p_acc  # 8p training performance result log
    test/output/devie_id/train_PraNet_bs16_8p_acc_loss   # 8p training accuracy result log

   模型训练脚本参数说明如下，以train_performance_1p.sh为例：

    
    ################基础配置参数，需要模型审视修改##################
    # 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
    # 网络名称，同目录名称
    Network="PraNet"
    # 训练batch_size
    batch_size=16
    # 训练使用的npu卡数
    export RANK_SIZE=1
    # 数据集路径,保持为空,不需要修改 train_path=./data/TrainDataset
    train_path=""

    # 训练epoch
    train_epochs=2
    # 指定训练所使用的npu device卡id=0 多卡id=1(用于输出日志)
    device_id=0
    # 加载数据进程数
    workers=128

    公共参数：
    --train_path=./data/TrainDataset                     //训练数据集路径 
    --addr=$(hostname -I |awk '{print $1}')              //主机地址
    --seed=49                                            //设置随机种子
    --workers=${workers}                                 //加载数据进程数
    --lr=1e-4                                            //初始学习率
    --world-size=1                                       //服务器台数
    --decay_epoch=50                                     //学利率衰退epoch数
    --device='npu'                                       //计算芯片类型
    --gpu=${ASCEND_DEVICE_ID}                            //计算芯片序号
    --dist-backend='hccl'                                //通信后端
    --epoch=${train_epochs}                              //训练epoch数
    --loss-scale=128                                     //loss-scale大小
    --amp                                                //是否开启混合精度
    --batchsize=${batch_size} > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &                        //日志路径
    
   

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | PT版本|精度 |  FPS | Epochs | AMP_Type |
| ------- |---- |----- | ---: | ------ | -------: |
| 1p-竞品V | 1.5|-     |  11 | 1      |        - |
| 1p-NPU  | 1.5|-     |  9.8 | 1      |       O2 |
| 1p-NPU  | 1.8|-     |  22.3 | 1      |       O2 |
| 8p-竞品V | 1.5|88.3 | 74.8 | 100    |        - |
| 8p-NPU  | 1.5|88.7 | 55.2 | 100    |       O2 |
| 8p-NPU  | 1.8|90.3 | 130.1 | 100    |       O2 |


# 版本说明

## 变更

2022.08.15：更新pytorch1.8版本，重新发布。

2020.07.08：首次发布。

## 已知问题

**_当前发行版本中存在的问题描述。_**

无。







