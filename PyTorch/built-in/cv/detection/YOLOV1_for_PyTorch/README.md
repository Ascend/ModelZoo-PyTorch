# YOLOV1 for PyTorch
- [YOLOV1 Dynamic for PyTorch](#yolov1-for-pytorch)
- [概述](#概述)
- [准备训练环境](#准备训练环境)
  - [准备环境](#准备环境)
  - [准备数据集](#准备数据集)
- [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)
  - [变更](#变更)
  - [已知问题](#已知问题)

# 概述
TOLOV1 是第一个利用回归方法进行目标检测的模型

- 参考实现：

  ```
  url=https://github.com/yjh0410/PyTorch_YOLO-Family
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/cv/detection/YOLOV1_for_PyTorch
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
  | 固件与驱动 | [22.0.3](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=5.1.RC1) |
  | PyTorch    | [1.8](https://gitee.com/ascend/pytorch/tree/master/)

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 1.安装好相应的cann包、pytorch和apex包，并设置好pytorch运行的环境变量。下载好模型源码并cd到YOLOV1_for_PyTorch目录
- 2.安装cocoapi
  ```
  git clone https://github.com/cocodataset/cocoapi.git
  cd cocoapi/PythonAPI
  python setup.py install
  ```
- 3.安装依赖
  ```
  pip install -r requirements.txt
  ```
## 准备数据集

1. 下载COCO数据集
2. 新建文件夹data
3. 将coco数据集放于data目录下
   ```
   ├── data
         ├──coco
              ├──annotations     
              ├──train2017
              ├──val2017                            
   ```

## 训练

### 1.训练方法
1p
```
bash ./test/train_full_1p.sh --data_path="***"
```
8p
```
bash ./test/train_full_8p.sh --data_path="***"
```  
   

# 训练结果展示

**表 2**  训练结果展示表

|   NAME  |  mAP  |  FPS  | AMP_Type |
| ------- | ----- | ----- | -------  |
| 1p-竞品 | -     |   80  |    O1    |
| 1p-NPU  | -     |  141  |    O1   |
| 8p-竞品 | 31.8  |   278  |    O1   |
| 8p-NPU  | 32.5  |  261  |    O1   |

# 版本说明

## 变更

2022.11.26：首次发布。

## 已知问题

**_当前发行版本中存在的问题描述。_**

无。