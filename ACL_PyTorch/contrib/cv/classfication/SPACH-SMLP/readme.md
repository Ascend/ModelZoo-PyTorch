
- [1. SPACH sMLP Onnx模型端到端推理指导](#1-spach-smlp-onnx模型端到端推理指导)
	- [1.1. 模型概述](#11-模型概述)
		- [1.1.1. 论文地址](#111-论文地址)
		- [1.1.2. 代码地址](#112-代码地址)
	- [1.2. 环境说明](#12-环境说明)
	- [1.3. 模型转换](#13-模型转换)
		- [1.3.1. pth转onnx模型[可选]](#131-pth转onnx模型可选)
		- [1.3.2. onnx转om模型](#132-onnx转om模型)
	- [1.4. 数据预处理](#14-数据预处理)
		- [1.4.1. 数据集获取](#141-数据集获取)
		- [1.4.2. 1.4.2.数据集预处理](#142-142数据集预处理)
	- [1.5. 离线推理](#15-离线推理)
		- [1.5.1. ais_infer工具概述](#151-ais_infer工具概述)
		- [1.5.2.  离线推理](#152--离线推理)
	- [1.6. 精度对比](#16-精度对比)
		- [1.6.1. 离线推理TopN精度](#161-离线推理topn精度)
		- [1.6.2. 开源TopN精度](#162-开源topn精度)
		- [1.6.3. 精度对比](#163-精度对比)
	- [1.7. 性能对比](#17-性能对比)
		- [1.7.1. npu性能数据](#171-npu性能数据)
		- [1.7.2. gpu，npu推理性能对比](#172-gpunpu推理性能对比)

# 1. SPACH sMLP Onnx模型端到端推理指导

## 1.1. 模型概述


### 1.1.1. 论文地址

[sMLP论文](https://arxiv.org/abs/2109.05422)

### 1.1.2. 代码地址

[sMLP代码](https://github.com/microsoft/SPACH)


## 1.2. 环境说明


```
CANN 5.0.4
torch==1.5.0+ascend.post5.20220315
torchvision
timm==0.3.2
einops==0.3.2
```

 **说明：**

- X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
- Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装
- jpg2bin.py、pth2onnx.py内部引用的datasets、models等模块需要配合[原仓库代码](https://github.com/microsoft/SPACH)一起使用。

## 1.3. 模型转换

- [sMLP预训练pth、onnx、om权重文件，提取码：xpo4](https://pan.baidu.com/s/1Na-LHL3ueS2V2ChB694Ztg)

### 1.3.1. pth转onnx模型[可选]

1. 下载pth权重文件
   [sMLP预训练pth、onnx、om权重文件，提取码：xpo4](https://pan.baidu.com/s/1Na-LHL3ueS2V2ChB694Ztg)

> **说明** pth文件的md5sum值为：061415304F38317C3850A587EF709D45 
> 文件下载后，放置与代码同一目录下。

2. sMLP模型代码在[sMLP代码](https://github.com/microsoft/SPACH)里，需要下载。
3. 调用pth2onnx脚本，生成onnx文件

> **说明**
> 注意目前ATC支持的onnx算子版本为11

4. 执行pth2onnx.py脚本，生成onnx模型文件

```
python3.7 pth2onnx.py
```

### 1.3.2. onnx转om模型

1. 设置环境变量

```
source env.sh
```

2. 使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)

```
# CANN安装目录
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
# 将atc日志打印到屏幕
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
# 设置日志级别
#export ASCEND_GLOBAL_LOG_LEVEL=0 #debug 0 --> info 1 --> warning 2 --> error 3
# 开启ge dump图
#export DUMP_GE_GRAPH=2
# 参考命令
atc --framework=5 --model=sMLPNet-T.onnx --output=sMLPNet-T-batch1-high --input_format=NCHW --input_shape="input:1,3,224,224" --soc_version=Ascend710 --op_precision_mode=op_precision.ini
```

若生成batch size为16的om模型，对应的命令为：

```
atc --framework=5 --model=sMLPNet-T.onnx --output=sMLPNet-T-batch16-high --input_format=NCHW --input_shape="input:16,3,224,224" --soc_version=Ascend710 --op_precision_mode=op_precision.ini
```

batch size为4、8、32的同上

## 1.4. 数据预处理


### 1.4.1. 数据集获取

> 该模型使用[ImageNet官网](http://www.image-net.org/)的5万张验证集进行测试，
> 图片与标签分别存放在/opt/npu/imagenet/val与/opt/npu/imagenet/val_label.txt。
> 如果放在其余位置，请修改jpg2bin.py文件中args的data_path参数。

### 1.4.2. 1.4.2.数据集预处理

1. 编写预处理脚本jpg2bin.py
   预处理方式有两种：不使用aipp的二进制输入，以及使用aipp的jpg输入。这里使用第一种，需要先用脚本仿照[SPACH-sMLP官方训练预处理方法处理数据](https://github.com/microsoft/SPACH)，以获得最佳精度；
2. 执行预处理脚本，生成数据集预处理后的bin文件

```
python3.7 img2bin.py 
```

## 1.5. 离线推理

### 1.5.1. ais_infer工具概述

ais_infer工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend710上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[tools: Ascend tools - ais_infer 推理工具使用文档 - Gitee.com](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer#%E4%BD%BF%E7%94%A8%E6%96%B9%E6%B3%95)
将获取的工具包并解压，将ais_infer工具放在当前目录下

### 1.5.2.  离线推理

1. 设置环境变量

``` 
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
```

2. 执行离线推理
   运行如下命令进行离线推理：
   
```
python3.7.5 ais_infer.py  --model /home/infname63/spach-smlp/sMLPNet-T-batch1-high.om  --batchsize 1 --output ./ --outfmt BIN --loop 100 
```

输出结果默认保存在当前目录中，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个BIN文件。

## 1.6. 精度对比

### 1.6.1. 离线推理TopN精度

- 离线推理，input目录放着转换为bin文件的imagenet val数据。
```
python3.7.5 ais_infer.py  --model /home/infname63/spach-smlp/sMLPNet-T-batch1-high.om --output ./ --input "/opt/npu/imagenet/val-bin-of-spach/" --outfmt NPY
```

- 精度统计
调用imagenet_acc_eval_ais_infer.py脚本与label比对，可以获得Accuracy Top1，Top5 准确率数据。

```
python imagenet_acc_eval_ais_infer.py ~/spach-smlp/ais_infer/2022_07_09-18_05_40/
```

查看输出的结果：

```
acc1:0.8125, acc5:0.9549
```


### 1.6.2. 开源TopN精度

GPU上使用[原仓库代码](https://github.com/microsoft/SPACH)对pth文件进行推理，参考连接：[推理pth，提取码：xpo4](https://pan.baidu.com/s/1Na-LHL3ueS2V2ChB694Ztg)

得到的结果是：

```
python main.py --eval --resume smlp_t.pth --model smlpnet_tiny --data-path /opt/gpu/imagenet/
Acc@1 81.74
Acc@5 95.79
```

### 1.6.3. 精度对比

将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，如下表所示，精度下降在1%范围之内，故精度达标。

| 模型                    | Acc@1  | Acc@5  |
| ----------------------- | ------ | ------ |
| pth模型推理结果（官方未提供acc@5，因此自行复现） | 81.74 | 95.79 |
| om模型离线推理结果      | 81.25 | 95.49 |

 **说明：**

> 没有遇到精度不达标的问题，故不需要进行精度调试

## 1.7. 性能对比


### 1.7.1. npu性能数据

对于使用ais_infer工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。
 **ais_infer工具在整个数据集上推理获得性能数据:**

1. batch1的性能，ais_infer工具在整个数据集上推理日志如下

```
infname63@d0c3e5f6b93c:~/spach-smlp/ais_infer$ python3.7.5 ais_infer.py  --model /home/infname63/spach-smlp/sMLPNet-T-batch1-high.om  --batchsize 1 --output ./ --outfmt BIN --loop 100  --output test
[INFO] load model /home/infname63/spach-smlp/sMLPNet-T-batch1-high.om success
[INFO] create model description success
[INFO] output path:test/2022_07_09-18_23_06
Inference Processing task: 100%|███████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 21.29it/s]
[INFO] -----------------Performance Summary------------------
[INFO] H2D_latency (ms): min = 0.2384185791015625, max = 0.2384185791015625, mean = 0.2384185791015625, median = 0.2384185791015625, percentile(99%) = 0.2384185791015625
[INFO] NPU_compute_time (ms): min = 5.828554630279541, max = 5.828554630279541, mean = 5.828554630279541, median = 5.828554630279541, percentile(99%) = 5.828554630279541
[INFO] D2H_latency (ms): min = 0.08606910705566406, max = 0.08606910705566406, mean = 0.08606910705566406, median = 0.08606910705566406, percentile(99%) = 0.08606910705566406
[INFO] throughput (1000*batchsize/NPU_compute_time): 171.5691219234638
[INFO] ------------------------------------------------------
[INFO] unload model success, model Id is 1
DestroyDevices begindestory device:0
aclrtDestroyContext successfully!
DestroyDevices successfully
```
即是batch1 710单卡吞吐率为171.569

2. batch16的性能，ais_infer工具在整个数据集上推理日志如下
   
```
python3.7.5 ais_infer.py  --model /home/infname63/spach-smlp/sMLPNet-T-batch16-high.om  --batchsize 16 --output ./ --outfmt BIN --loop 100  --output test
[INFO] load model /home/infname63/spach-smlp/sMLPNet-T-batch16-high.om success
[INFO] create model description success
[INFO] output path:test/2022_07_09-18_25_15
Inference Processing task: 100%|███████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.79it/s]
[INFO] -----------------Performance Summary------------------
[INFO] H2D_latency (ms): min = 1.8732547760009766, max = 1.8732547760009766, mean = 1.8732547760009766, median = 1.8732547760009766, percentile(99%) = 1.8732547760009766
[INFO] NPU_compute_time (ms): min = 55.18075942993164, max = 55.18075942993164, mean = 55.18075942993164, median = 55.18075942993164, percentile(99%) = 55.18075942993164
[INFO] D2H_latency (ms): min = 0.10800361633300781, max = 0.10800361633300781, mean = 0.10800361633300781, median = 0.10800361633300781, percentile(99%) = 0.10800361633300781
[INFO] throughput (1000*batchsize/NPU_compute_time): 289.95613988090815
[INFO] ------------------------------------------------------
[INFO] unload model success, model Id is 1
DestroyDevices begindestory device:0
aclrtDestroyContext successfully!
DestroyDevices successfully
```
即是batch16 710单卡吞吐率为289.95613988090815

### 1.7.2. gpu，npu推理性能对比

| batchsize | GPU-t4      | ascend-710  |
|-----------|-------------|-------------|
| 1         | 195.5718619 | 171.5691219 |
| 4         | 367.6943724 | 273.5174541 |
| 8         | 395.153443  | 298.7036461 |
| 16        | 389.3891458 | 289.9561399 |
| 32        | 398.4738452 | 272.9734707 |
| 64        | 407.7238181 | 257.5456754 |
| best      | 407.7238181 | 298.7036461 |

