# MMseg-Swin for PyTorch
-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述
MMsegmentation仓，使用swin-transfermer作为backbone做分割任务。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmsegmentation.git
  tags=v0.26.0
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
  | NPU固件与驱动 | [1.0.16](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/)

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 请先安装好昇腾Pytorch与Apex后再进行依赖的安装。  
- 安装依赖。

  ```
  pip3 install -r requirements.txt
  pip3 install -e .
  ```

- Build MMCV。

  ```
  export MMCV_WITH_OPS=1
  export MAX_JOBS=8

  cd mmcv
  python3 setup.py build_ext
  python3 setup.py develop
  pip3 list | grep mmcv
  # 安装opencv-python-headless, 规避cv2引入错误
  pip3 uninstall opencv-python
  pip3 install opencv-python-headless
  ```

## 准备数据集

1. 获取数据集。

   ADE20K数据集是2016年MIT开放的场景理解的数据集，可用于实例分割，语义分割和零部件分割。利用图像信息进行场景理解 scene understanding和 scene parsing。数据集目录结构参考如下所示。

   ```
    ADEChallengeData2016
    ├── annotations
    │   ├── training
    │   │   ├── ADE_train_00000001.png
    │   │   ├── ADE_train_00000002.png
    │   │   ├── ADE_train_00000003.png
    │   │   ├── ADE_train_00000004.png
    │   │   ├── ADE_train_00000005.png
    └── validation
    │       ├── ADE_val_00000001.png
    │       ├── ADE_val_00000002.png
    │       ├── ADE_val_00000003.png
    │       ├── ADE_val_00000004.png
    │       ├── ADE_val_00000005.png
    ├── images
    │   ├── training
    │   │   ├── ADE_train_00000001.jpg
    │   │   ├── ADE_train_00000002.jpg
    │   │   ├── ADE_train_00000003.jpg
    │   │   ├── ADE_train_00000004.jpg
    │   │   ├── ADE_train_00000005.jpg
    │   └── validation
    │       ├── ADE_val_00000001.jpg
    │       ├── ADE_val_00000002.jpg
    │       ├── ADE_val_00000003.jpg
    │       ├── ADE_val_00000004.jpg
    │       ├── ADE_val_00000005.jpg
    ├── objectInfo150.txt
    └── sceneCategories.txt
   ```
    在源码包根目录下构建数据集软链接：
    ```
    mkdir data
    cd data
    mkdir ade
    cd ade
    ln -s train_path/ADEChallengeData2016
    ```

   > **说明：** 
   >数据集路径'train_path'以用户自行定义的路径为准。


## 获取预训练模型
     
1. 模型训练启动后，预训练模型会自动下载。


# 开始训练

## 训练模型
1. 进入解压后的源码包根目录。

    ```
    cd /${模型文件夹名称} 
    ```

2. 运行训练脚本。

    ```bash
    # training accuracy
    bash ./test/train_full.sh ${CONFIG_FILE} ${NPU_NUM}

    # training performance
    bash ./test/train_performance.sh ${CONFIG_FILE} ${NPU_NUM}
    ```

3. 示例：使用8卡训练"upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K"配置，得到精度和性能数据。

    ```bash
    # training accuracy
    bash ./test/train_full.sh configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py 8

    # training performance
    bash ./test/train_performance.sh configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py 8
    ```
4. 目前已支持的配置：

    ```bash
    configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py
    configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py
    configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py
    configs/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_1K.py
    ```

日志路径:
    
    test/output/devie_id/train_${device_id}.log           # training detail log
    test/output/devie_id/MMseg_bs2_p_acc  # training performance result log
    test/output/devie_id/train_MMseg_bs2_p_acc_loss   # training accuracy result log

   模型训练脚本参数说明如下，以train_performance.sh为例：

    
    ################基础配置参数，需要模型审视修改##################
    # 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
    # 网络名称，同目录名称
    Network="MMseg"
    # 训练batch_size
    batch_size=2
    # 训练使用的npu卡数
    GPUS=$2
    # 数据集路径,保持为空,不需要修改（设置软链接后，可不输入train_path=""）
    train_path=""

    公共参数：
    --nnodes=$NNODES \                            //多级多卡训练指定机器数
    --node_rank=$NODE_RANK \                      //使用卡号  
    --master_addr=$MASTER_ADDR \                  //DDP主机地址
    --nproc_per_node=$GPUS \                      //训练使用的npu卡数
    --master_port=$PORT \                         //DDP主端口
    ./tools/train.py \
    $CONFIG \                                     //模型配置
    --launcher pytorch ${@:3}
    
   

# 训练结果展示

精度指标为mIoU

**表 2**  MMseg-Swin-Tiny训练结果展示表

| NAME    | PT版本|精度 |  FPS | Iters | AMP_Type |
| ------- |---- |----- | ---: | ------ | -------: |
| 1p-竞品V | 1.5|-     |  6.12 | 1000      |        - |
| 1p-NPU  | 1.5|-     |  6.47 | 1000      |       O1 |
| 8p-竞品V | 1.5|44.41 | 45.72| 160000    |        - |
| 8p-NPU  | 1.5|44.07 | 35.40 | 160000    |       O1 |

**表 3**  MMseg-Swin-Small训练结果展示表

| NAME    | PT版本|精度 |  FPS | Iters | AMP_Type |
| ------- |---- |----- | ---: | ------ | -------: |
| 1p-竞品V | 1.5|-     |  4.55 | 1000      |        - |
| 1p-NPU  | 1.5|-     |  4.47 | 1000      |       O1 |
| 8p-竞品V | 1.5|44.92 | 29.1 | 48000    |        - |
| 8p-NPU  | 1.5|44.19 | 26.45 | 48000    |       O1 |

**表 4**  MMseg-Swin-Base224训练结果展示表

| NAME    | PT版本|精度 |  FPS | Iters | AMP_Type |
| ------- |---- |----- | ---: | ------ | -------: |
| 1p-竞品V | 1.5|-     |  4.87 | 1000      |        - |
| 1p-NPU  | 1.5|-     |  3.78 | 1000      |       O1 |
| 8p-竞品V | 1.5|47.97 | 28.57 | 160000    |        - |
| 8p-NPU  | 1.5|47.04 | 22.91 | 160000    |       O1 |

**表 5**  MMseg-Swin-Base384训练结果展示表

| NAME    | PT版本|精度 |  FPS | Iters | AMP_Type |
| ------- |---- |----- | ---: | ------ | -------: |
| 1p-竞品V | 1.5|-     |  4.79 | 1000      |        - |
| 1p-NPU  | 1.5|-     |  3.48 | 1000      |       O1 |
| 8p-竞品V | 1.5|48.35 | 27.59 | 160000    |        - |
| 8p-NPU  | 1.5|47.52 | 22.36 | 160000    |       O1 |

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md

# 版本说明

## 变更

2022.11.14：首次发布。

## 已知问题


无。