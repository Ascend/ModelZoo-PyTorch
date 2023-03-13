# TSM(mmaction2) for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

这是一个基于UCF101/Sthv2数据库训练的TSM模型，基于open-mmlabmmaction2三方库迁移。
TSN模型从视频中采样N帧图像并通过最简单直接地对N帧图像分类结果进行平均的方式进行时序信息融合，取得了当时State-of-the-art的性能，并得到大规模的应用。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmaction2
  commit_id=f2dcc05d5cbe18cdabc8b4248d339dce4a8ac5db
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/video
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 Pytorch 版本和已知已知三方库依赖如下所示。

  **表 1**  版本支持表
 
  | Torch_Version |
  |---------------|
  | Pytorch 1.5   |
  | Pytorch 1.8   |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  

- 安装依赖：
  
  在模型源码包根目录下执行命令，安装模型需要的依赖。

   ```
   pip install -r requirements.txt
  ```

- 安装 MMCV：
  ```
  cd ${TSM}
  git clone -b v1.3.9 https://github.com/open-mmlab/mmcv.git
  cd mmcv
  pip install -e ./
  ```

  替换mmcv文件：
  ```
  /bin/cp -f mmcv_need/base_runner.py mmcv/mmcv/runner/base_runner.py
  /bin/cp -f mmcv_need/builder.py mmcv/mmcv/runner/optimizer/builder.py
  /bin/cp -f mmcv_need/checkpoint.py mmcv/mmcv/runner/hooks/checkpoint.py
  /bin/cp -f mmcv_need/data_parallel.py mmcv/mmcv/parallel/data_parallel.py
  /bin/cp -f mmcv_need/dist_utils.py mmcv/mmcv/runner/dist_utils.py
  /bin/cp -f mmcv_need/distributed.py mmcv/mmcv/parallel/distributed.py
  /bin/cp -f mmcv_need/epoch_based_runner.py mmcv/mmcv/runner/epoch_based_runner.py
  /bin/cp -f mmcv_need/iter_timer.py mmcv/mmcv/runner/hooks/iter_timer.py
  /bin/cp -f mmcv_need/optimizer.py mmcv/mmcv/runner/hooks/optimizer.py
  /bin/cp -f mmcv_need/test.py mmcv/mmcv/engine/test.py
  /bin/cp -f mmcv_need/transformer.py mmcv/mmcv/cnn/bricks/transformer.py
  ```

## 准备数据集
1. 获取数据集

    ucf101请参照datasets/README.MD进行数据下载和预处理，somethingv2数据集根据GPU官网指导手动下载。
      
2. 数据集预处理

   ```
   mkdir ${TSM}/data
   ```
  - ucf101预处理后进行数据软连接：
   
    ```
    ln -s ${ucf101} data/ucf101
    ```
  - somethingv2请先进行解压和软连接，再执行预处理脚本：
    ```
    cd ${sthv2}
    cat 20bn-something-something-v2-?? | tar zx #将原始数据进行解压
    mv ${sthv2}/20bn-something-something-v2 ${sthv2}/videos #重命名原始数据集

    cd ${TSM}
    ln -s ${sthv2} data/sthv2
    bash dataset/sthv2_prepare.sh
    ```
    数据集的路径需写到数据集的一级目录，预处理后的文件目录结构与datasets/README.MD中Step 6保持一致。

# 开始训练

## 训练模型
1. 进入解压后的源码包根目录。
    ```
    cd /${模型文件名称} 
    ```
2. 运行训练脚本。

    该模型支持单机单卡训练和单机8卡训练。

  - 单机单卡训练

    启动单卡训练。

    ```
    bash test/train_performance_1p.sh --config=config # 单卡性能
    ```

  - 单机8卡训练

    启动8卡训练。

    ```
    bash test/train_full_8p.sh --config=config # 8卡精度 
    bash test/train_performance_8p.sh --config=config # 8卡性能 
    ```
    --config传参在[ufc101, sthv2]中选填。

模型训练脚本参数说明如下。

```
公共参数：
--data_root                     //是否额外传入数据集路径
--gpu-ids                       //设备号
--cfg-options                   //修改配置参数
--resume-from                   //断点续训地址
```
训练完成后，权重文件保存在当前路径result/下，并输出模型训练精度和性能信息。
 
# 训练结果展示

**表 2**  ucf101训练结果展示表

| NAME   | ACC    | FPS      | Epoch | AMP_Type | Torch_Version |
|--------|--------|----------|-------|----------|---------------|
| 1p-NPU |    -   | 38.18 | 1     | O2       | 1.5           |
| 8p-NPU | 0.9392 | 295.889 | 32     | O2      | 1.5          |
| 1p-竞品V |    -   | 57.23 | 1     | O2      | 1.5           |
| 8p-竞品V | 0.9387 | 410.492 | 32     | O2      | 1.5           |


**表 3**  sthv2训练结果展示表

| NAME   | ACC    | FPS      | Epoch | AMP_Type | Torch_Version |
|--------|--------|----------|-------|----------|---------------|
| 8p-NPU | 0.5911 |    152   |   50  |  O2      | 1.8           |
| 8p-竞品V | 0.5895|    116   |   50  |  O2      | 1.8           |

# 版本说明

## 变更

2023.02.24：更新内容，支持sthv2数据集训练。

2022.03.18：首次发布，支持ucf101数据集训练。

## FAQ
   无。










