# ICNet Onnx模型端到端推理指导
[TOC]



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[ICNet论文](https://arxiv.org/abs/1704.08545)  
我们专注于本文中实时语义分割的挑战性任务。 它发现了许多实际应用，但在减少像素级标签推理的大部分计算方面存在根本性的困难。 我们提出了一种图像级联网络 (ICNet)，它在适当的标签指导下结合了多分辨率分支来应对这一挑战。 我们对我们的框架进行了深入分析，并引入了级联特征融合单元以快速实现高质量分割。 我们的系统在单个 GPU 卡上产生实时推理，并在具有挑战性的数据集（如 Cityscapes、CamVid 和 COCO-Stuff）上评估出质量不错的结果。

### 1.2 代码地址
[ICNet代码](https://github.com/liminn/ICNet-pytorch)  
branch:master  
commit_id:da394fc44f4fbaff1b47ab83ce7121a96f375b03  
备注：commit_id是指基于该次提交时的模型代码做推理，通常选择稳定版本的最后一次提交，或代码仓最新的一次提交

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.1.RC1
Pytorch == 1.8.0
torchvision == 0.9.0
onnx == 1.9.0
```

### 2.2 python第三方库

```
numpy == 1.20.2
Pillow == 8.2.0
opencv-python == 4.5.2.52
sympy == 1.4
decorator == 4.4.2
requests == 2.22.0
tqdm == 4.61.0
PyYAML == 5.4.1
```

**说明：** 

>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 准备数据集

> 以下均已310P为例总结分析，310正常跑通



### 3.1 获取原始数据集。

 该模型使用cityscapes数据集的500张验证集进行测试。用户自行获取数据集cityscapes，解压并上传到modelzoo源码包根目录下。

###  3.2数据预处理 

 将原始数据（.jpeg）转化为二进制文件（.bin）。执行预处理脚本，生成数据集预处理后的bin文件。 

```
python3.7 pre_dataset.py ./cityscapes/ ./pre_dataset_bin
```

### 3.3 生成数据集info文件 

 使用benchmark推理需要输入二进制数据集的info文件，用于获取数据集。使用get_info.py脚本，输入已经得到的二进制文件，输出生成二进制数据集的info文件。运行get_info.py脚本。 

```
python3.7 get_info.py bin ./pre_dataset_bin ./icnet_pre_bin_1024_2048.info 1024 2048
```

第一个参数为生成的数据集文件格式，第二个参数为预处理后的数据文件路径，第三个参数为生成的数据集文件保存的路径，第四个和第五个参数分别为模型输入的宽度和高度。

运行成功后，在当前目录中生成icnet_pre_bin_1024_2048.info。

## 4、模型推理

### 4.1 模型转换 

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

####  a. 获取权重文件 

 从源码包获取A800-9000训练的pth文件rankid0_icnet_resnet50_192_0.687_best_model.pth。

####  b. 导出onnx文件 

#####  i.获取DnCNN模型开源代码。 

```
git clone https://github.com/liminn/ICNet-pytorch.git
cd ICNet-pytorch
git reset da394fc44f4fbaff1b47ab83ce7121a96f375b03 --hard
```

##### ii. 修改模型代码。 

```
patch -p1 < ../icnet.diff
cd ..
cp -rf ./ICNet-pytorch/utils ./
```

##### iii. 修改resnetv1b.py脚本。 

```
vi ICNet-pytorch/models/base_models/resnetv1b.py
```

 由于权重文件在训练时注释了下面两行代码，转onnx模型时手动添加注释。 

```
# 将下面两行代码注释。
# self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
# self.fc = nn.Linear(512 * block.expansion, num_classes)
```

##### iv. 执行ICNet_pth2onnx.py脚本，生成onnx模型文件，执行如下命令在当前目录生成ICNet.onnx模型文件。 

```
python3.7 ICNet_pth2onnx.py rankid0_icnet_resnet50_192_0.687_best_model.pth ICNet.onnx
```

####  c.使用ATC工具将ONNX模型转OM模型 

##### i. 设置环境变量 

```
source env.sh
```

该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 (推理)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。 

##### ii. 使用atc将onnx模型转换为om模型文件。 

```
atc --framework=5 --model=ICNet.onnx --output=ICNet_bs1 --out_nodes="Resize_317:0" --input_format=NCHW --input_shape="actual_input_1: 1,3,1024,2048" --log=debug --soc_version=${chip_name}
```

${chip_name}可通过npu-smi info 指令查看

- 参数说明：

  - --model：为ONNX模型文件。

  - --framework：5代表ONNX模型。

  - --output：输出的OM模型。

  - --input_format：输入数据的格式。

  - --input_shape：输入数据的shape。

  - --log：日志级别。

  - --soc_version：处理器型号。

    运行成功后生成ICNet_bs1.om用于二进制输入推理的模型文件。 

### 4.2开始推理验证 

#### a. 使用Benchmark工具进行推理 

##### i. 设置环境变量。 

```
source env.sh
```

 脚本中环境变量install_path请修改为CANN toolkit包的实际安装路径。 

##### ii. 增加benchmark.*{arch}*可执行权限。 

```
chmod u+x benchmark.x86_64
```

##### iii. 执行离线推理 

```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./ICNet_bs1.om -input_text_path=./icnet_pre_bin_1024_2048.info -input_width=1024 -input_height=2048 -output_binary=True -useDvpp=False
```

icnet_pre_bin_1024_2048.info为处理后的数据集信息。

执行./benchmark*.x86_64*工具请选择与运行环境架构相同的命令。参数详情请参见《[CANN 推理benchmark工具用户指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。推理后的输出默认在当前目录result下。

#### b. 精度验证。 

 调用evaluate.py脚本推理，结果保存在icnet_bs*x*.log中。 

```
python3.7 -u evaluate.py ./cityscapes ./result/dumpOutput_device0 ./out >icnet_bs1.log
```

 第一个参数为数据集路径，第二个参数为benchmark输出目录，第三个为输出存储路径。 

## 5精度对比

### 5.1 离线推理精度统计
由于模型结够太大，bs8报内存不足，可以支持batch1，batch4的离线推理，batch1与batch4精度相同    

调用evaluate.py脚本推理，结果保存在icnet_bsx.log中。
```
python3.7 -u evaluate.py ./cityscapes ./result/dumpOutput_device0 ./out >icnet_bs1.log
```
第一个参数为数据集路径，第二个参数为benchmark输出目录，第三个为输出存储路径，推理日志存储在icnet_bs1.log中
查看icnet_bs1.log最后一行推理结果：

```
Evaluate: Average mIoU: 0.689, Average pixAcc: 0.950, Average time: 21.313
```

### 5.3 精度对比
将得到的om离线模型推理精度与之前推理精度对比，推理精度与之前推理精度一致，精度达标。  

## 6 性能对比

### 6.1性能数据

#### 6.1.1 npu性能数据

由于模型结够太大，bs32报内存不足，因此仅测试batch1，batch4，batch8，batch16的性能，这里用batch1做示例   

benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device。为快速获取性能数据，也可以使用benchmark纯推理功能测得性能数据，模型的测试脚本使用benchmark工具在整个数据集上推理得到batch1，batch4，batch8，batch16性能数据为准。    

benchmark纯推理功能测得性能数据   

batch1的性能：
 测试npu性能要确保device空闲，使用npu-smi info命令可查看device是否在运行其它推理任务

```
./benchmark.x86_64 -round=20 -om_path=ICNet_bs1.om -device_id=0 -batch_size=1
```
执行20次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果
```
ave_throughputRate:20.6958samples/s, ave_latency:48.5792ms
```

batch4的性能：
```
./benchmark.x86_64 -round=20 -om_path=ICNet_bs4.om -device_id=0 -batch_size=4
```
```
ave_throughputRate:34.784samples/s, ave_latency:28.752ms
```

batch8的性能：

```
./benchmark.x86_64 -round=20 -om_path=ICNet_bs8.om -device_id=0 -batch_size=8
```

```
ave_throughputRate:33.6264samples/s, ave_latency:29.7413ms
```

batch16的性能：

```
./benchmark.x86_64 -round=20 -om_path=ICNet_bs16.om -device_id=0 -batch_size=16
```

```
ave_throughputRate:33.8414samples/s, ave_latency:29.5532ms
```

#### 6.1.2 GPU数据

只测试T4的bs1、bs4、bs8、bs16:

```
trtexec --onnx=ICNet.onnx --fp16 --shapes=image:1,3,1024,2048 --threads
```

bs1:

```
GPU Compute Time: min = 13.2834 ms ，max = 14.6467 ms ，mean = 13.8817 ms，median = 13.8707 ms，percentile( 99%) = 14.6328 ms
```

bs4:

```
GPU Compute Time: min = 13.2086 ms，max = 14.7099 ms,mean = 13.8078 ms， median = 13.7419 ms,percentile(99%) = 14.5562 ms
```

bs8:

```
GPU Compute Time: min = 13.2487 ms，max = 14.719 ms，mean = 13.7947 ms， median = 13.7551 ms，percentile(99%) = 14.571 ms
```

bs16:

```
GPU Compute Time: min = 13.0369 ms， max = 14.425 ms，mean = 13.5858 ms, median = 13.5699 ms，percentile(99%)= 14.2744 ms
```

### 6.2 性能优化

####  6.2.1 性能分析

- profiling性能分析方法  
  

		CANN C20及以后的版本profiling使用方法
		新建/home/HwHiAiUser/test/run文件，内容如下：
		#! /bin/bash
		export install_path=/usr/local/Ascend/ascend-toolkit/latest
		export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
		export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
		export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
		export ASCEND_OPP_PATH=${install_path}/opp
		./benchmark -round=50 -om_path=/home/HwHiAiUser/test/efficientnet-b0_bs1.om -device_id=0 -batch_size=1
		
		然后执行如下命令：
		chmod 777 /home/HwHiAiUser/test/run
		cd /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/toolkit/tools/profiler/bin
		./msprof --output=/home/HwHiAiUser/test --application=/home/HwHiAiUser/test/run --sys-hardware-mem=on --sys-cpu-profiling=on --sys-profiling=on --sys-pid-profiling=on --sys-io-profiling=on --dvpp-profiling=on
		cd /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/toolkit/tools/profiler/profiler_tool/analysis/msprof/
		python3.7 msprof.py import -dir /home/HwHiAiUser/test/生成的profiling目录
		python3.7 msprof.py export summary -dir /home/HwHiAiUser/test/生成的profiling目录

- 性能调优测试版本：CANN 5.1.RC1

- 性能优化过程主，结合profiling分析，性能有提升（但是很小）:


以bs1为例：

```
ave_throughputRate:20.7127samples/s, ave_latency:48.5689ms
```

#### 6.2.1 AOE优化方法  

1、设置环境变量 

```	

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export LD_LIBRARY_PATH=/usr/local/Ascend/ascendtoolkit/5.1.RC1/tools/aoe/lib64:$LD_LIBRARY_PATH

export TUNE_BANK_PATH=/home/zsl/icnet/aoe

export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH

export PATH=/usr/local/python3.7.5/bin:$PATH
```

 2、参考命令： 
```
aoe --framework 5 --model ICNet.onnx  --job_type 2 --output ICNet_bs1-aoe  --input_shape "actual_input_1: 1,3,1024,2048"  --log error --op_select_implmode high_performance
```
- 不同的batch_size,添加的shape不一样，shape大小为[*，3,1024,2048 ] ,以本模型为例，只测试batch1和batch4,batch8,batch16因此添加的shape为[1,3,1024,2048],[4,3,1024,2048],[8,3,1024,2048],[16,3,1024,2048]

重新转换生成om文件。



纯推理测试结果：

bs1:

```
ave_throughputRate = 20.9682samples/s, ave_latency = 47.7961ms
```

用生成的om文件，测得bs1推理吞吐量率为20.9682，优化前吞吐量率为20.6958，性能有所提升（但是很小）。

bs4,bs8,bs16经过AOE调优之后，性能都有所提升但是很小。但是当我将AOE调优之后的om文件转到其他310p机子上纯推理时，发现性能竟然不如优化之前。
#### 6.2.3 Auto Tune 

1.设置环境变量

```
export TUNE_BANK_PATH=/home/zsl/icnet/autotune
```
2.执行调优

```
atc --framework=5 --model=ICNet.onnx --output=ICNet_bs1 --out_nodes="Resize_317:0" --input_format=NCHW --input_shape="actual_input_1: 1,3,1024,2048" --log=debug --soc_version=Ascend310  --auto_tune_mode="RL,GA"
```

"RL,GA"：同时进行RL与GA的调优，RL与GA顺序不区分，Auto Tune工具会自动根据算子特点选择使用RL调优模式还是GA调优模式。

纯推理测试结果：

```
bs1:
ave_throughputRate = 21.5788samples/s, ave_latency = 46.4836ms
```
用生成的om文件，测得bs1推理吞吐量率为21.5788，未优化前推理精度为20.6958，精度未下降。bs4,bs8,bs16经过Auto Tune调优之后，性能都有所提升但是很小.但是当我将AOE调优之后的om文件转到其他310p机子上纯推理时，发现性能竟然不如优化之前。
#### 6.3 总结
优化方案共包括两种：  
（1）AOE 
（2）Auto Tune  

主要对比两种优化方案对推理性能的提升，如下：

| optimization point | bs1(FPS) |
| :----------------: | :------: |
|       优化前       | 20.6958  |
|        AOE         | 20.9682  |
|     Auto Tune      | 21.5788  |

由以上数据可以看出：  
(1) bs1的吞吐率由最初的20.6958提升至20.9682（AOE),21.5788(Auto Tune)，虽有所提升，但是很小。  

结论：
- 因为关键算子性能差，310P/T4性能暂时无法达标。(310与310p的精度、性能均已达标)
- 最终精度测试，测得bs1推理精度为310：Average mIoU: 0.686, Average pixAcc: 0.940，310P:Average mIoU: 0.689, Average pixAcc: 0.950以上两种优化方案，虽然提升了性能，但是提升的很小

| Batch Size | 310         | 310p        | aoe后的310p | T4            | 310p(autotune后) |
| ---------- | ----------- | ----------- | ----------- | ------------- | ---------------- |
| bs1        | **11.7012** | **20.6958** | **20.9682** | **72.0372**   | **21.5788**      |
| bs4        | **13.27**   | **34.784**  | **34.1967** | **289.6913**  | **34.7779**      |
| bs8        | **13.3353** | **33.6264** | **35.3837** | **579.9454**  | **37.6133**      |
| bs16       |             | **33.8411** | **39.9992** | **1177.7175** | **40.4716**      |

| Model | Batch Size | 310p(FPS/Card) | T4 (FPS/Card) | 310p/T4    |
| ----- | ---------- | -------------- | ------------- | ---------- |
| ICNet | 1          | **20.6958**    | **72.0372**   | **0.2872** |
| ICNet | 4          | **34.784**     | **289.6913**  | **0.1200** |
| ICNet | 8          | **35.3837**    | **579.9454**  | **0.0610** |
| ICNet | 16         | **39.9992**    | **1177.7175** | **0.0339** |