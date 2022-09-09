# SSD for PyTorch
- [SSD Dynamic for PyTorch](#ssd-for-pytorch)
- [概述](#概述)
- [准备训练环境](#准备训练环境)
  - [准备环境](#准备环境)
  - [准备数据集](#准备数据集)
- [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)
  - [变更](#变更)
  - [已知问题](#已知问题)

# 概述
SSD 是利用不同尺度的特征图进行目标的检测的模型

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection/tree/v2.25.0
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/cv/detection/SSD_for_PyTorch
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
  | 固件与驱动 | [22.0.2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=5.1.RC1) |
  | PyTorch    | [1.8](https://gitee.com/ascend/pytorch/tree/master/)

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 1.安装好相应的cann包、pytorch和apex包，并设置好pytorch运行的环境变量。下载好模型源码并cd到SSD_for_PyTorch目录
- 2.安装mmcv。
  ```
  git clone -b v1.4.8 --depth=1 https://github.com/open-mmlab/mmcv.git
  export MMCV_WITH_OPS=1
  export MAX_JOBS=8
  cd mmcv
  python3.7 setup.py build_ext
  python3.7 setup.py develop
  export mmcv_path=mmcv安装路径
  ```
- 2.安装MMDET
  cd到SSD_for_PyTorch目录
  ```
  pip3 install -r requirements/build.txt
  pip3 install -v -e .
  ```
- 3.替换mmcv文件
  ```
  /bin/cp -f mmcv_need/builder.py ${mmcv_path}/mmcv/runner/optimizer/
  /bin/cp -f mmcv_need/dist_utils.py ${mmcv_path}/mmcv/runner/
  /bin/cp -f mmcv_need/distributed.py ${mmcv_path}/mmcv/parallel/
  /bin/cp -f mmcv_need/optimizer.py ${mmcv_path}/mmcv/runner/hooks/
  ```


## 准备数据集

1. 下载COCO数据集
2. 新建文件夹data
3. 将coco数据集放于data目录下
   ```
   ├── data
         ├──coco
              ├──annotations     
              ├──result
              ├──train2017
              ├──val2017                            
   ```

## 训练

### 1.训练方法

1p
```
bash run_1p.sh
```
8p
```
bash run_8p.sh
```  
   

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | mAP |  s/iter | Epochs | AMP_Type |
| ------- | ----- | ---: | ------ | -------: |
| 1p-竞品 | -     |  0.290 | 1   |       O1 |
| 1p-NPU  | -     |  0.230 | 1   |       O1 |
| 8p-竞品 | 20.1 | 0.339 | 120    |       O1 |
| 8p-NPU  | 20.1 | 0.257 | 120    |       O1 |

# 版本说明

## 变更

2022.09.01：首次发布。

## 已知问题

**_当前发行版本中存在的问题描述。_**

无。