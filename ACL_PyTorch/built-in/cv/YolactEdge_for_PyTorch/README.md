# yolact_edge模型推理指导

- [yolact_edge模型推理指导](#yolact_edge模型推理指导)
  - [1 文件说明](#1-文件说明)
  - [2 环境准备](#2-环境准备)
    - [2.1 安装依赖](#21-安装依赖)
    - [2.2 文件下载](#22-文件下载)
    - [2.3 文件拷贝](#23-文件拷贝)
    - [2.4 设置环境变量](#24-设置环境变量)
  - [3 端到端推理步骤](#3-端到端推理步骤)
    - [3.1 pth导出onnx](#31-pth导出onnx)
    - [3.2 利用ATC工具转换为om模型](#32-利用atc工具转换为om模型)
    - [3.3 om模型推理](#33-om模型推理)
    - [3.4 纯推理性能获取](#34-纯推理性能获取)
  - [4 评测结果](#4-评测结果)

------

## 1 文件说明
```
yolact_edge_for_Pytorch
  ├── pth2onnx.py       pytorch模型导出onnx模型
  ├── atc.sh            onnx模型转om
  ├── yolact_edge.diff  补丁文件
  └── acl_net.py        PyACL推理工具代码
```

## 2 环境准备

### 2.1 安装依赖

根据pytorch官网教程安装1.10.0版本的PyTorch
```shell
pip install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu
pip install -r requirements.txt
pip install git+https://github.com/haotian-liu/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
```

### 2.2 文件下载
- [yolact_edge_Pytorch源码下载](https://github.com/haotian-liu/yolact_edge)

  ```shell
  git clone git@github.com:haotian-liu/yolact_edge.git
  cd yolact_edge
  git reset a9a00281b33b3ac90253a4939773308a8f95e21d --hard
  git apply yolact_edge.diff
  ```

- 权重下载

  创建 `weights` 目录，并将下载的权重文件 `yolact_edge_resnet50_54_800000.pth` 拷贝到 `weights` 目录下。
  可参见[yolact_edge_Pytorch主页](https://github.com/haotian-liu/yolact_edge)说明下载权重

- 数据集下载

  om推理采用COCO2017数据集的验证集进行精度评估。将下载好的数据集拷贝到 `data/coco` 目录下，data目录中的文件结构如下所示。数据集下载可以网页搜索
  ```shell
  data
    └── coco
      ├── annotations
      ├── images
    ├── scripts
    ├── yolact_edge_example_1.gif
    ├── yolact_edge_example_2.gif
    ├── yolact_edge_example_3.gif
  ```

### 2.3 文件拷贝
拷贝env.sh，pth2onnx.py，atc.sh，acl_net.py文件到yolact_edge目录下。


### 2.4 设置环境变量
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 3 端到端推理步骤

### 3.1 pth导出onnx
```python
python3.7 pth2onnx.py \
        --config=yolact_edge_resnet50_config \
        --trained_model=./weights/yolact_edge_resnet50_54_800000.pth
```

### 3.2 利用ATC工具转换为om模型
```shell
bash atc.sh yolact_edge.onnx yolact_edge
```

### 3.3 om模型推理
```python
python3.7 eval.py \
        --config=yolact_edge_resnet50_config \
        --trained_model=./weights/yolact_edge_resnet50_54_800000.pth \
        --cuda=False \
        --disable_tensorrt
```

### 3.4 纯推理性能获取

下载 [benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer) 并拷贝到当前目录
```shell
./benchmark.${arch} -device_id=0 -batch_size=1 -om_path=./yolact_edge.om -round=20
```

## 4 评测结果

| 模型            | pth精度  | 310离线推理精度 | 基准性能 | 310性能 |
| --------------- | -------- | --------------- | -------- | ------- |
| yolact_edge | [mAP:27.0](https://github.com/haotian-liu/yolact_edge) | mAP:27.6        | 167fps   | 157fps  |
