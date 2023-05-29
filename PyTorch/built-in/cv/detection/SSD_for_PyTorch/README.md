# SSD for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述
SSD 是利用不同尺度的特征图进行目标的检测的模型。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection/tree/v2.25.0
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

- 安装mmcv。

  在源码包根目录下执行以下安装命令。
  
  ```shell
  git clone -b v1.4.8 --depth=1 https://github.com/open-mmlab/mmcv.git
  export MMCV_WITH_OPS=1
  export MAX_JOBS=8
  cd mmcv
  python3 setup.py build_ext
  python3 setup.py develop
  ```
- 安装MMDET。

  在源码包根目录下执行以下安装命令。

  ```shell
  pip3 install -r requirements/build.txt
  pip3 install -v -e .
  ```
- 替换mmcv文件。

  在源码包根目录下执行以下替换命令。

  ```shell
  cp -f mmcv_need/builder.py ${mmcv_path}/mmcv/runner/optimizer/
  cp -f mmcv_need/dist_utils.py ${mmcv_path}/mmcv/runner/
  cp -f mmcv_need/distributed.py ${mmcv_path}/mmcv/parallel/
  cp -f mmcv_need/optimizer.py ${mmcv_path}/mmcv/runner/hooks/
  ```
  > **说明：** 
  >mmcv_path为mmcv实际文件路径。


## 准备数据集

1. 获取数据集。

   请用户自行下载COCO数据集，并在源码包根目录下新建文件夹**data**，将coco数据集放于data目录下，数据集目录结构参考如下所示。

   ```
   ├── data
         ├──coco
              ├──annotations     
              ├──result
              ├──train2017
              ├──val2017                            
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

     ```shell
     bash ./test/train_performance_1p.sh  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```shell
     bash ./test/train_full_8p.sh  # 8卡精度
     ```

   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --seed                              //随机数种子设置
   --gpu-ids                           //训练卡id设置
   --opt-level                         //混合精度类型
   --work-dir                          //日志和模型保存目录     
   --auto-resume                       //自动加载最后的checkpoint进行恢复
   --diff-seed                         //是否在不同rangk上设置不同的随机数种子
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

| NAME    | mAP |  s/iter | Epochs | AMP_Type | Torch_Version |
| :-----: | :---: | :--: | :----: | :------: | :------: |
| 1p-竞品V | -     |  0.290 | 1   |       O1 | 1.5 |
| 8p-竞品V | 20.1 | 0.339 | 120    |       O1 | 1.5 |
| 1p-NPU  | -     |  0.230 | 1   |       O1 | 1.8 |
| 8p-NPU  | 20.1 | 0.257 | 120    |       O1 | 1.8 |

# 版本说明

## 变更

2023.02.22：更新readme，重新发布。

2022.09.01：首次发布。

## FAQ

无。