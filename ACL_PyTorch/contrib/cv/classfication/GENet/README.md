## 1、环境准备
### 1.下载pth模型
点击[链接](https://pan.baidu.com/s/1U_Se8EwZbfToc6Yi8wH-sw)下载原始pth模型，提取码hwst<br/>
原始pth模型由对应的[训练代码](https://gitee.com/fleurrr/modelzoo/tree/master/contrib/PyTorch/Research/cv/image_classification/GENET_for_PyTorch)训练得到
### 2.安装必要环境
```
pip3.7 install -r requirements.txt  
```
### 3.获取，安装开源模型代码
```
git clone https://github.com/BayesWatch/pytorch-GENet.git 
cd pytorch-GENet/
git reset 3fbf99fb6934186004ffb5ea5c0732e0e976d5b2 --hard
cd ../
```
### 4.下载数据集
在官方[链接](http://www.cs.toronto.edu/~kriz/cifar.html)下载cifar10数据集

### 5.获取benchmark工具 
[下载](https://gitee.com/ascend/cann-benchmark/tree/master/infer)benchmark工具，将benchmarck.x86_64放到推理目录 

## 2、om模型转换
开始推理工作前，先设置环境变量
```
bash test/env.sh
```
运行`pthtar2onnx.py`文件以转换得到对应的onnx模型。其中${model_path}指的是模型文件路径，如/home/HwHiAiUser/model/genet.pth.tar
```
python3.7 pthtar2onnx.py ${model_path}

```
将生成onnx模型放在推理代码主目录，运行以下命令以转换om模型，模型转换中使用了autotune工具以提升om模型性能
```
bash test/onnx2om.sh
```
该指令会生成genet_bs16_tuned，genet_bs1_tuned两个模型

## 3、准备数据集
下载cifar10数据集，置于${datasets_path}，运行以下命令以前处理数据集。注意，第一个参数为数据集存放目录（例：若数据集路径为/home/HwHiAiUser/dataset/cifar-10-batches-py/，则数据集存放目录为/home/HwHiAiUser/dataset/）
```
python3.7 preprocess.py ${datasets_path} ./prep_dataset
```
运行以下指令将数据转换为bin格式，生成标签文件./prep_dataset/val_label.txt，生成数据集信息文件genet_prep_bin.info
```
python3.7 get_info.py bin ./prep_dataset ./genet_prep_bin.info 32 32
```
## 4、310推理
运行以下指令生成推理结果文件
```
bash test/infer_bin.sh
```
该指令会在result目录下生成模型的推理结果文件以及推理性能文件，推理性能文件参考如下
```
[e2e] throughputRate: 135.162, latency: 73985.3
[data read] throughputRate: 138.155, moduleLatency: 7.23827
[preprocess] throughputRate: 137.422, moduleLatency: 7.27687
[infer] throughputRate: 138.1, Interface throughputRate: 554.397, moduleLatency: 4.00654
[post] throughputRate: 138.098, moduleLatency: 7.24122
```
## 5、精度测试
参考以下指令生成精度信息文件
```
python3.7 cifar10_acc_eval.py result/dumpOutput_device0/ ./prep_dataset/val_label.txt ./ result_bs1.json
```
参考以下指令获取310上推理的精度以及性能信息
```
python3.7 test/parse.py result_bs1.json
python3.7 test/parse.py result/perf_vision_batchsize_1_device_0.txt
```
将onnx模型置于装有gpu的设备，参考一下指令获取onnx模型在gpu上推理的性能信息
```
bash test/perf_g.sh
```

## 6、执行入口及模型信息

执行入口：
```
bash test/eval_acc_perf.sh --datasets_path='dataset path'
```
|  GENET模型|  gpu吞吐率| 310吞吐率 |  精度|
|--|--|--|--|
|  bs1|  1829.937fps| 2217.588fps|Error@1 5.76 Error@5 0.15|
|  bs16| 5652.112fps| 6919.96fps|Error@1 5.78 Error@5 0.15 |