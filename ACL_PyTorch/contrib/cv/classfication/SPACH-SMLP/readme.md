
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
		- [1.4.2. 数据集预处理](#142-数据集预处理)
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
- smlp_preprocess.py、smlp_pth2onnx.py内部引用的datasets、models等模块需要配合[原仓库代码](https://github.com/microsoft/SPACH)一起使用。

## 1.3. 模型转换


### 1.3.1. pth转onnx模型[可选]

1. 下载pth权重文件
   sMLP预训练[pth](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/sMLP/smlp_t.pth)、[onnx权重文件](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/sMLP/sMLPNet-T.onnx)

> **说明** pth文件的md5sum值为：061415304F38317C3850A587EF709D45 
> 文件下载后，放置与代码同一目录下。

1. sMLP模型代码在[sMLP代码](https://github.com/microsoft/SPACH)里，需要下载。
2. 调用smlp_pth2onnx脚本，生成onnx文件


3. 执行smlp_pth2onnx.py脚本，生成onnx模型文件

```
python3.7 smlp_pth2onnx.py --pth_path smlp_t.pth --onnx_path sMLPNet-T.onnx
```

参数说明：

	usage: pytorch pth convert to onnx [-h] [--model_name MODEL] [--pth_path PTH_PATH] [--onnx_path ONNX_PATH] [--opset_version OPSET_VERSION]

	options:
	-h, --help            show this help message and exit
	--model_name MODEL    Name of model to convert
	--pth_path PTH_PATH   path to checkpoint
	--onnx_path ONNX_PATH
							path to checkpoint
	--opset_version OPSET_VERSION
							opset version

### 1.3.2. onnx转om模型

1. 设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
```

2. 使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)


${chip_name}可通过`npu-smi info`指令查看，例：310P3

![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```
atc --framework=5 --model=sMLPNet-T.onnx --output=sMLPNet-T-batch1-high --input_format=NCHW --input_shape="input:1,3,224,224" --soc_version=Ascend${chip_name} --op_precision_mode=op_precision.ini
```

参数说明：

	--model：为ONNX模型文件。  
	--framework：5代表ONNX模型。  
	--input_format：输入数据的格式。  
	--input_shape：输入数据的shape。  
	--output：输出的OM模型。  
	--log：日志级别。  
	--soc_version：处理器型号。  
	--insert_op_config：插入算子的配置文件路径与文件名，例如aipp预处理算子。  
	--enable_small_channel：Set enable small channel. 0(default): disable; 1: enable  

执行后在当前目录下生成om模型文件：sMLPNet-T-batch1-high.om。

点击这里可以直接下载，已生成的，batch size为[1](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/sMLP/sMLPNet-T-batch1-high.om)、[4](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/sMLP/sMLPNet-T-batch4-high.om)、[8](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/sMLP/sMLPNet-T-batch8-high.om)、[16](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/sMLP/sMLPNet-T-batch16-high.om)、[32](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/sMLP/sMLPNet-T-batch32-high.om)、[64](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/sMLP/sMLPNet-T-batch64-high.om)的om模型

## 1.4. 数据预处理


### 1.4.1. 数据集获取

> 该模型使用[ImageNet官网](http://www.image-net.org/)的5万张验证集进行测试，
> 如果放在其余位置，请修改smlp_preprocess.py文件中args的data_path参数。

### 1.4.2. 数据集预处理

1. 调用预处理脚本smlp_preprocess.py
   需要先调用该脚本，仿照[SPACH-sMLP官方训练预处理方法处理数据](https://github.com/microsoft/SPACH)，以获得最佳精度；
2. 执行预处理脚本，生成数据集预处理后的bin文件

```
python3.7 smlp_preprocess.py --save_dir imagenet-val-bin --data_root /opt/npu/imagenet/
```

参数说明：

	usage: validate model [-h] [--batch_size BATCH_SIZE] [--data_root DATA_ROOT] --save_dir SAVE_DIR

	options:
	-h, --help            show this help message and exit
	--batch_size BATCH_SIZE
	--data_root DATA_ROOT
							dataset path
	--save_dir SAVE_DIR   dir path to save bin

## 1.5. 离线推理

### 1.5.1. ais_infer工具概述

ais_infer工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310p上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[tools: Ascend tools - ais_infer 推理工具使用文档 - Gitee.com](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer#%E4%BD%BF%E7%94%A8%E6%96%B9%E6%B3%95)
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
python imagenet_acc_eval_ais_infer.py ${result_dir_path}
```

参数说明：
    ${result_dir_path}参数：为ais_infer.py运行后生成的存放推理结果的目录的路径。 例如本例中为~/spach-smlp/ais_infer/2022_07_09-18_05_40/

    
	

查看输出的结果：

```
acc1:0.8125, acc5:0.9549
```


### 1.6.2. 开源TopN精度

GPU上使用[原仓库代码](https://github.com/microsoft/SPACH)对pth文件进行推理

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

## 1.7. 性能对比


### 1.7.1. npu性能数据

对于使用ais_infer工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。
 **ais_infer工具在整个数据集上推理获得性能数据:**

1. batch1的性能，ais_infer工具在整个数据集上推理日志如下

```
infname63@d0c3e5f6b93c:~/spach-smlp/ais_infer$ python3.7.5 ais_infer.py  --model /home/infname63/spach-smlp/sMLPNet-T-batch1-high.om  --batchsize 1 --output ./ --outfmt BIN --loop 100  --output test
[INFO] -----------------Performance Summary------------------
[INFO] H2D_latency (ms): min = 0.2384185791015625, max = 0.2384185791015625, mean = 0.2384185791015625, median = 0.2384185791015625, percentile(99%) = 0.2384185791015625
[INFO] NPU_compute_time (ms): min = 5.828554630279541, max = 5.828554630279541, mean = 5.828554630279541, median = 5.828554630279541, percentile(99%) = 5.828554630279541
[INFO] D2H_latency (ms): min = 0.08606910705566406, max = 0.08606910705566406, mean = 0.08606910705566406, median = 0.08606910705566406, percentile(99%) = 0.08606910705566406
[INFO] throughput (1000*batchsize/NPU_compute_time): 171.5691219234638
```
即是batch1 310p单卡吞吐率为171.569

2. batch16的性能，ais_infer工具在整个数据集上推理日志如下
   
```
python3.7.5 ais_infer.py  --model /home/infname63/spach-smlp/sMLPNet-T-batch16-high.om  --batchsize 16 --output ./ --outfmt BIN --loop 100  --output test
[INFO] -----------------Performance Summary------------------
[INFO] H2D_latency (ms): min = 1.8732547760009766, max = 1.8732547760009766, mean = 1.8732547760009766, median = 1.8732547760009766, percentile(99%) = 1.8732547760009766
[INFO] NPU_compute_time (ms): min = 55.18075942993164, max = 55.18075942993164, mean = 55.18075942993164, median = 55.18075942993164, percentile(99%) = 55.18075942993164
[INFO] D2H_latency (ms): min = 0.10800361633300781, max = 0.10800361633300781, mean = 0.10800361633300781, median = 0.10800361633300781, percentile(99%) = 0.10800361633300781
[INFO] throughput (1000*batchsize/NPU_compute_time): 289.95613988090815
```

即是batch16 310p单卡吞吐率为289.95613988090815

### 1.7.2. gpu，npu推理性能对比

| batchsize | ascend-310p | GPU-t4 |
|-----------|------------|--------|
| 1         | 171.6      | 177.7  |
| 4         | 273.5      | 341.5  |
| 8         | 298.7      | 359.0  |
| 16        | 290.0      | 363.7  |
| 32        | 273.0      | 371.0  |
| 64        | 257.5      | 359.1  |
| best      | 298.7      | 371.0  |

> **说明：**
> NPU和GPU的推理性能（吞吐率）对比为： 0.805    
