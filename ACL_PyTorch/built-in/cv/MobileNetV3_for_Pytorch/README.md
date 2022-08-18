# MobileNetV3模型推理指导

- [MobileNetV3模型推理指导](#MobileNetV3模型推理指导)
	- [1 环境准备](#1-环境准备)
		- [1.1 下载pytorch源码](#11-下载pytorch源码)
		- [1.2 准备以下文件，放到pytorch源码根目录](#12-准备以下文件放到pytorch源码根目录)
		- [1.3 安装依赖](#13-安装依赖)
	- [2 推理步骤](#2-推理步骤)
		- [2.1 设置环境变量](#21-设置环境变量)
		- [2.2 pth导出onnx模型](#22-pth导出onnx模型)
		- [2.3 onnx导出om模型](#23-onnx导出om模型)
		- [2.4 模型推理](#24-模型推理)
	- [3 端到端推理Demo](#3-端到端推理demo)

------


## 1 环境准备

### 1.1 下载pytorch源码
```shell
git clone https://github.com/xiaolai-sqlai/mobilenetv3
```

### 1.2 准备以下文件，放到pytorch源码根目录
（1）该代码仓文件  
（2）`ILSVRC2012`，推理采用`ImageNet 2012`数据集的验证集进行精度评估，在`pytorch`源码根目录下新建`imagenet`文件夹，数据集放到`imagenet`里，文件结构如下：
```
imagenet
└── val
  ├── n01440764
    ├── ILSVRC2012_val_00000293.jpeg
    ├── ILSVRC2012_val_00002138.jpeg
	……
	└── ILSVRC2012_val_00048969.jpeg
  ├── n01443537
  ……
  └── n15075141
└── val_label.txt
```

### 1.3 安装依赖
安装 [om推理接口工具](https://gitee.com/peng-ao/pyacl)

## 2 推理步骤
### 2.1 设置环境变量
根据实际安装的CANN包路径修改`/usr/local/Ascend/ascend-toolkit`。
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2.2 pth导出onnx模型
运行`pth2onnx.py`导出onnx模型，默认保存在`output`文件夹下，可设置参数`--dynamic`支持导出动态batch的onnx，`--simplify`简化导出的onnx。
```python
python3 pth2onnx.py --pth=mbv3_small.pth.tar --onnx=mbv3_small.onnx --batch=1 --dynamic --simplify
```

### 2.3 onnx导出om模型
运行`atc.sh`导出om模型，默认保存在`output`文件夹下。
```shell
bash atc.sh [soc] output mbv3_small 1  # Ascend310P3
```
其中，`[soc]`可通过`npu-smi info`命令查看，如果所示`soc`为`Ascend310P3`。
![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

### 2.4 模型推理
运行`val.py`推理`pth/om`模型，结果默认保存在`output`文件夹下。
```shell
python3 val.py --dataset=imagenet --checkpoint=output/mbv3_small_bs16.om --batch=1
```

## 3 端到端推理Demo
提供`run.sh`，作为实现端到端推理的样例参考，默认参数设置见`run.sh`，其中，`soc`为必选参数。
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash run.sh [soc]  # Ascend310P3
```
### 3.1 精度结果
| 精度指标 | pytorch |  om  |
|  :---:  |  :---:  |  :-:  |
|   Top1   |  65.074 | 65.104 |
|   Top5   |  85.436 | 85.432 |