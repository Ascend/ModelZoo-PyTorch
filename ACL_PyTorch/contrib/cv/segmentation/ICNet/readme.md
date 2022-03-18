# ICNet Onnx模型端到端推理指导
-   [1 模型概述](#1-模型概述)
	-   [1.1 论文地址](#11-论文地址)
	-   [1.2 代码地址](#12-代码地址)
-   [2 环境说明](#2-环境说明)
	-   [2.1 深度学习框架](#21-深度学习框架)
	-   [2.2 python第三方库](#22-python第三方库)
-   [3 模型转换](#3-模型转换)
	-   [3.1 pth转onnx模型](#31-pth转onnx模型)
	-   [3.2 onnx转om模型](#32-onnx转om模型)
-   [4 数据集预处理](#4-数据集预处理)
	-   [4.1 数据集获取](#41-数据集获取)
	-   [4.2 数据集预处理](#42-数据集预处理)
	-   [4.3 生成数据集信息文件](#43-生成数据集信息文件)
-   [5 离线推理](#5-离线推理)
	-   [5.1 benchmark工具概述](#51-benchmark工具概述)
	-   [5.2 离线推理](#52-离线推理)
-   [6 精度对比](#6-精度对比)
	-   [6.1 离线推理精度统计](#61-离线推理精度统计)
	-   [6.2 在线推理精度](#62-在线推理精度)
	-   [6.3 精度对比](#63-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)
	-   [7.2 性能优化](#73-性能优化)
	    - [7.2.1 优化TransData，修改five_2_four.py](#731-优化TransData，修改five_2_four.py)
		- [7.2.2 输出节点由float32改为float16](#732-输出节点由float32改为float16)
		- [7.2.3 模型中Resize节点的mode由双线性为最近邻](#733-模型中Resize节点的mode由双线性为最近邻)
		- [7.2.4 将PadV3D进行算子融合](#734-将PadV3D进行算子融合)
		- [7.2.5 优化FrameworkOP框架](#735-优化FrameworkOP框架)



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
CANN 5.0.2.alpha003
torch == 1.8.0
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

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型
1.下载开源模型代码，安装必要的依赖库，并修改模型代码后安装   
```
git clone https://github.com/liminn/ICNet-pytorch.git
cd ICNet-pytorch
# pip3.7 install -r requirements.txt
git reset da394fc44f4fbaff1b47ab83ce7121a96f375b03 --hard
patch -p1 < ../icnet.diff
cd ..
cp -rf ./ICNet-pytorch/utils ./
```

2.下载pth权重文件  

- [官方ICNet pth权重文件](https://github.com/liminn/ICNet-pytorch/)  
- 获取A800-9000训练的pth文件
```  
wget https://ascend-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/instance_segmentation/ICNet/%E6%8E%A8%E7%90%86/rankid0_icnet_resnet50_192_0.687_best_model.pth
```
md5sum值：bf3a0da8c3e11dfba4fac3b98f9a6874  

3.编写pth2onnx脚本ICNet_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件  
```
python3.7 ICNet_pth2onnx.py rankid0_icnet_resnet50_192_0.687_best_model.pth ICNet.onnx
```

 **模型转换要点：**  
- 开源仓中model部分应用adaptive_avg_pool2d算子进行训练和在线推理，onnx转换过程不支持该算子，参考同等语义替换，用avg_pool2d算子实现onnx转换

``` 
# models/icnet.py脚本
if self.train_mode:
	x = F.adaptive_avg_pool2d(input, output_size=bin_size)
else:
	inputsz = np.array(input.shape[2:])
	outputsz = np.array([bin_size, bin_size])
	stridesz = np.floor(inputsz / outputsz).astype(np.int32)
	kernelsz = inputsz - (outputsz - 1) * stridesz
	print("========avg para kernelsz, stridesz======:", kernelsz, stridesz)
	x = F.avg_pool2d(input, kernel_size=list(kernelsz), stride=list(stridesz))
```
- 使用CANN 5.0.1版本测试时，在onnx转om过程会报错，提示
```
not support ksize[2] 12 * ksize[3] *ksize 22 > 255 or strides[2] 10 > 63 strides[3] 21 > 63
```
在蓝区用CANN 5.0.2.alpha003版本测试，此问题已经规避  

- NPU A800-9000训练过程，将如下两行代码注释训练的，如果用官方提供的pth权重文件推理，需要将这两行代码打开
``` 
# models/base_models/resnetv1b.py脚本
# self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
# self.fc = nn.Linear(512 * block.expansion, num_classes)
``` 

- 有些pytorch算子onnx还不支持，根据开源社区提供的方法等价替换这些算子，如果不能完全等价替换而且npu已经支持该算子，则需要修改模型代码将该算子封装为自定义算子，然后导出包含自定义算子的onnx    

例如，pytorch代码的adaptive_avg_pool2d目前onnx还不支持，所以导出onnx时报错，解决方案是尝试使用avg_pool2d替换adaptive_avg_pool2d，但当input的最后两维不是output的整数倍时，adaptive_avg_pool2d不能完全等价替换为avg_pool2d，而npu有adaptive_avg_pool2d算子的实现，所以解决方案变为将adaptive_avg_pool2d改为自定义算子导出onnx，自定义算子不需要具体实现代码(因此导出的onnx不能使用onnxruntime进行推理，还需要将pytorch的_check_onnx_proto(proto)改为pass去除导出onnx时进行检查)，只要自定义算子返回的输出shape与原算子输出的shape保持一致即可，相当于onnx只包含这个算子的声明（数据类型与属性需要与npu版算子对应），在onnx转为om时，atc工具的onnx插件如果支持该算子，atc工具会根据这个声明找到该算子npu的实现。     

在CANN包安装目录的opp下搜索AdaptiveAvgPool2d，查看npu的adaptive_avg_pool2d声明：
```
REG_OP(AdaptiveAvgPool2d)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(output_size, ListInt)
    .OP_END_FACTORY_REG(AdaptiveAvgPool2d)
```
修改模型代码，将adaptive_avg_pool2d改为自定义算子，然后导出onnx，其中output_size_i代表int64类型的算子属性：
```
class AdaptiveAvgPoolOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, output_size):
        out = torch.randn(x.shape[0], x.shape[1], output_size[0], output_size[1]).to(x.dtype)
        return out
    @staticmethod
    def symbolic(g, x, output_size):
        out = g.op('AdaptiveAvgPool2d', x, output_size_i = output_size)
        return out
def adaptive_avg_pool_op(x, output_size):
    out = AdaptiveAvgPoolOp.apply(x, output_size)
    return out
x = F.adaptive_avg_pool2d(input, output_size=bin_size)替换为x = adaptive_avg_pool_op(input, (bin_size, bin_size))
```

### 3.2 onnx转om模型

1.设置环境变量
```
source env.sh
```
- 根据实际情况修改env.sh中的install_path=/usr/local/Ascend/ascend-toolkit/latest变量  
- 执行脚本前先执行指令 dos2unix *

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN 5.0.1 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)
```
atc --framework=5 --model=ICNet.onnx --output=ICNet_bs1 --out_nodes="Resize_317:0" --input_format=NCHW --input_shape="actual_input_1: 1,3,1024,2048" --log=debug --soc_version=Ascend310
```
- 说明  
    - input_shape参数可通过Netron工具查看输入节点的名称和shape, 与pth转onnx步骤中的参数一致  
    - out_nodes为指定输出节点, 通过Netron可以看到onnx文件有四个输出, 以自测转换的onnx为例，794,788,782,756   
  
            INPUTS
                 actual_input_1  name: actual_input_1
                                 type: float32[1,3,1024,2048]
            OUTPUTS                    
                 794   name: 794
                       type: float32[1,19,1024,2048]
                 788   name: 788
                       type: float32[1,19,256,512]
                 782   name: 782
                       type: float32[1,19,128,256]
                 756   name: 756
                       type: float32[1,19,64,128]

    其分别对应evaluate.py脚本outputs = model(image)中的outputs[0], outputs[1], outputs[2], outputs[3]，脚本中仅需要outputs[0]的数据做推理使用self.metric.update(outputs[0], mask)。  
    
    因此在转om的时候, 仅存储outputs[0]节点的数据即可, 即889的节点输出，通过Netron工具可以看到889节点对应的输出名为Resize_317  
  
        NODE PROPERTIES
              type   Resize
              name   Resize_317
        
        OUTPUTS
              Y      name: 794


## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[cityscapes数据集](https://www.cityscapes-dataset.com)的500张验证集进行测试

### 4.2 数据集预处理
1.预处理脚本pre_dataset.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 pre_dataset.py /opt/npu/cityscapes/ ./pre_dataset_bin
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 get_info.py bin ./pre_dataset_bin ./icnet_pre_bin_1024_2048.info 1024 2048
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息  

## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN 5.0.1 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)
### 5.2 离线推理
1.设置环境变量
```
source env.sh
```
2.执行离线推理  
- benchmark工具区分arm64和x86_64, 对应分别为./benchmark.aarch64和./benchmark.x86_64, 示例中均以x86_64环境为例
- 将benchmark工具去相应路径获取后放到env.sh同级目录下，加上执行权限chmod +x benchmark.XX    
- 该模型支持多batch, 由于模型结够太大，bs8报内存不足，可支持batch1，batch4的离线推理
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./ICNet_bs1.om -input_text_path=./icnet_pre_bin_1024_2048.info -input_width=1024 -input_height=2048 -output_binary=True -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_devicex，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[离线推理TopN精度](#61-离线推理TopN精度)**  
-   **[开源TopN精度](#62-开源TopN精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度统计
由于模型结够太大，bs8报内存不足，可支持batch1，batch4的离线推理，batch1与batch4精度相同    

调用evaluate.py脚本推理，结果保存在icnet_bsx.log中。
```
python3.7 -u evaluate.py /opt/npu/cityscapes ./result/dumpOutput_device0 ./out >icnet_bs1.log
```
第一个参数为数据集路径，第二个参数为benchmark输出目录，第三个为输出存储路径，推理日志存储在icnet_bs1.log中
查看icnet_bs1.log最后一行推理结果：
```
semantic_segmentation INFO: Evaluate: Average mIoU: 0.681, Average pixAcc: 0.940, Average time: 21.608
```

### 6.2 在线推理精度
使用NPU A800-9000环境训练生成的pth文件，下载[官方ICNet源码](https://github.com/liminn/ICNet-pytorch/), 进行在线推理

	Evaluation
	First, modify the configuration in the configs/icnet.yaml file:
	
	### 4.Test
	test:
	  ckpt_path: "./ckpt/icnet_resnet50_197_0.710_best_model.pth"  # set the pretrained model path correctly
	Then, run: python3 evaluate.py

使用NPU A800-9000环境训练生成的pth文件,在线推理精度参考如下：  
```
semantic_segmentation INFO: Evaluate: Average mIoU: 0.680, Average pixAcc: 0.952, Average time: 0.625
```
### 6.3 精度对比
将得到的om离线模型推理精度与在线推理精度对比，推理精度与在线推理精度一致，精度达标。  
 **精度调试:**  
由于onnx不支持adaptive_avg_pool2d算子转换，使用avg_pool2d替换adaptive_avg_pool2d后，官方的pth转换成的om，精度不达标，使用onnxruntime测试onnx离线推理精度与om一致，说明了精度下降是因为input的最后两维不是output的整数倍时，avg_pool2d不能完全等价替换adaptive_avg_pool2d  

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  
-   **[性能优化](#73-性能优化)**  

### 7.1 npu性能数据
由于模型结够太大，bs8报内存不足，因此仅测试batch1，batch4的性能，这里用batch1做示例   

benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device。为快速获取性能数据，也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准。这里给出两种方式，模型的测试脚本使用benchmark工具在整个数据集上推理得到bs1与bs4的性能数据为准。    

1.benchmark工具在整个数据集上推理获得性能数据  

以batch1为例，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  
```
[e2e] throughputRate: 1.05117, latency: 475659
[data read] throughputRate: 1.57547, moduleLatency: 634.731
[preprocess] throughputRate: 1.20557, moduleLatency: 829.485
[infer] throughputRate: 1.05475, Interface throughputRate: 1.22385, moduleLatency:                                                                                                          945.98
[post] throughputRate: 1.05411, moduleLatency: 948.665
```
Interface throughputRate: 1.22385，1.22385乘以4，是310单卡吞吐率  

2.benchmark纯推理功能测得性能数据  

batch1的性能：
 测试npu性能要确保device空闲，使用npu-smi info命令可查看device是否在运行其它推理任务
```
./benchmark.x86_64 -round=20 -om_path=ICNet_bs1.om -device_id=0 -batch_size=1
```
执行20次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果
```
[INFO] Dataset number: 19 finished cost 817.067ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_ICNet_bs1_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 1.22321samples/s, ave_latency: 817.799ms
----------------------------------------------------------------

```
batch4的性能：
```
./benchmark.x86_64 -round=20 -om_path=ICNet_bs4.om -device_id=0 -batch_size=4
```
```
[INFO] Dataset number: 19 finished cost 3254.82ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_ICNet_bs4_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 1.22732samples/s, ave_latency: 814.763ms
----------------------------------------------------------------
```

### 7.2 性能优化

 **性能优化**  
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
  
- 性能调优测试版本：CANN 5.0.2.alpha003
- 性能优化过程主要对trans_Data算子进行优化，结合profiling分析，性能有提升:

| optimization point     | Op Type       | Output Shapes    | Output Data Types  | Output Formats   | aicore_time(us)  | ave_throughputRate  |
| :-------------------:  | :-----------: | :------------:   | :----------------: | :--------------: | :--------------: | :-----------------: |
| initial                | TransData     | "1,19,1024,2048" |  DT_FLOAT          |  NCHW            | 558231.5199      |  1.223            |
| five_2_four.py         | TransData     | "1,19,1024,2048" |  DT_FLOAT          |  NCHW            | 272430.3662      |  1.87               |
| output_node: float16   | TransData     | "1,19,1024,2048" |  DT_FLOAT16        |  NCHW            | 5483.2           |  3.7                |


#### 7.3.1 five_2_four.py优化方法  
  在环境变量env.sh中export install_path=/usr/local/Ascend/ascend-toolkit/latest路径下查找five_2_four.py文件，路径一般为
```	
/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/opp/op_impl/built-in/ai_core/tbe/impl/five_2_four.py
```

修改five_2_four.py文件，将TransData算子的output shape加入five_2_four函数行中，示例如下：
```
from impl import trans_data_negative_target_ntc

@util.check_input_type(dict, dict, str, str, str)
def five_2_four(src, dst, src_format, dst_format, kernel_name='five_2_four'):  
	...
	elif dst_format.lower() == "nhwc" and dst_shape in [[10000, 63, 63, 1], [10000, 127, 127, 1], [16, 19, 19, 486],
                                                        [16, 10, 10, 486], [16, 38, 38, 324], [16, 5, 5, 486],
                                                        [16, 3, 3, 324], [8, 19, 19, 486], [8, 10, 10, 486],
                                                        [8, 38, 38, 324], [8, 5, 5, 486], [8, 3, 3, 324],
                                                        [100, 28, 28, 91]]:
        trans_data_negative_target_tc.trans_data_negative_target_tc(src, dst, src_format, dst_format, kernel_name)
    elif dst_format.lower() == "nchw" and dst_shape in [[2560, 512, 4, 26], [2560, 512, 1, 26], [2560, 256, 8, 25],
                                                        [16, 240, 7, 7], [16, 120, 14, 14], [1,19,1024,2048], [4,19,1024,2048]]:
        print("=================================")
        print("ntc dst shape:", dst_shape)
        print("=================================")
        trans_data_negative_target_ntc.trans_data_negative_target_ntc(src, dst, src_format, dst_format, kernel_name)
	...
```
- 不同的batch_size,添加的shape不一样，shape大小为[*，19,1024,2048 ] ,以本模型为例，只测试batch1和batch4,因此添加的shape为[1,19,1024,2048],[4,19,1024,2048]

修改完成后，重新转换生成om文件，atc转换过程会打印添加的日志，如下：  
```
ATC start working now, please wait for a moment.
=================================
ntc dst shape: [1, 19, 1024, 2048]
=================================
=================================
ntc dst shape: [1, 19, 1024, 2048]
=================================
ATC run success, welcome to the next use.
W11001: High-priority service of op[PartitionedCall_AvgPool_45_2] is invalid, low-priority service is used. It can work normally but may affect performance.
W11001: High-priority service of op[PartitionedCall_AvgPool_52_6] is invalid, low-priority service is used. It can work normally but may affect performance.
```
纯推理测试结果：
```
bs1:
[INFO] Dataset number: 19 finished cost 535.227ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_ICNet_bs1_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 1.86946samples/s, ave_latency: 535.118ms
----------------------------------------------------------------

bs4:
[INFO] Dataset number: 19 finished cost 1470.12ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_ICNet_bs4_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 2.72028samples/s, ave_latency: 367.701ms
----------------------------------------------------------------
```

用生成的om文件做精度后处理，测得bs1推理精度为68.1，未优化前推理精度为68.1，精度未下降。
```
Sample: 500, validation pixAcc: 94.912, mIoU: 71.137, time: 21.355s
Evaluate: Average mIoU: 0.681, Average pixAcc: 0.943, Average time: 22.067
```
测得bs4推理精度为68.1，未优化前推理精度为68.1，精度未下降。
```
Sample: 500, validation pixAcc: 94.584, mIoU: 71.141, time: 22.438s
Evaluate: Average mIoU: 0.681, Average pixAcc: 0.940, Average time: 21.533
```

#### 7.3.2 output_node输出节点类型更改为float16

atc转换时指定输出节点类型为float16
```
atc --framework=5 --model=./ICNet.onnx --output=ICNet_bs1 --out_nodes="Resize_317:0" --output_type=FP16 --input_format=NCHW --input_shape="actual_input_1: 1,3,1024,2048" --log=debug --soc_version=Ascend310
```
纯推理测试结果：
```
bs1:
[INFO] Dataset number: 19 finished cost 272.321ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_ICNet_bs1_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 3.66939samples/s, ave_latency: 272.739ms
----------------------------------------------------------------

bs4:
[INFO] Dataset number: 19 finished cost 1077.64ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_ICNet_bs4_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 3.70877samples/s, ave_latency: 269.711ms
----------------------------------------------------------------
```
精度后处理时，由于输出节点shape改为float16，因此在解析离线推理的结果时，需要按照2字节读取，修改evaluate.py脚本：
```
def file2tensor(self, annotation_file):

	filepath = annotation_file + '_1.bin'
	size = os.path.getsize(filepath)  
	res = []
	L = int(size/2)  # float32 -> 4, float16 -> 2
	binfile = open(filepath, 'rb')  
	for i in range(L):
		data = binfile.read(2)   # float32 -> 4, float16 -> 2 
		num = struct.unpack('h', data)  # float32 -> f, float16 -> h 
		res.append(num[0])
	binfile.close()
```
用生成的om文件做精度后处理，测得bs1推理精度为68.1，未优化前推理精度为68.1，精度未下降。
```
Sample: 500, validation pixAcc: 94.912, mIoU: 71.137, time: 21.355s
Evaluate: Average mIoU: 0.681, Average pixAcc: 0.943, Average time: 22.067
```
测得bs4推理精度为20.5，未优化前推理精度为68.1，精度下降47.6，百分比为69.9%,下降幅度太大。
```
Sample: 500, validation pixAcc: 61.418, mIoU: 19.786, time: 22.948s
Evaluate: Average mIoU: 0.204, Average pixAcc: 0.621, Average time: 21.665
```

#### 7.3.3 模型中Resize节点的mode由双线性改为最近邻

该方法主要是将模型中的Resizebilinear算子替换为 nearest，以此提升推理性能。  
修改方法：
将生成的onnx文件下载到本地，使用Netron工具查看，搜索模型中的Resize节点，以自测模型为例，模型中Resize节点包括：  
Resize_7、Resize_67、Resize_212、Resize_229、Resize_246、Resize_263、Resize_283、Resize_307、Resize_314、Resize_371  

将查找到的所有节点的mode改为nearest，修改方法参照resize.py脚本
```
wget https://ascend-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/instance_segmentation/ICNet/%E6%8E%A8%E7%90%86/resize.py
```
本模型中Resize节点如下，如果换成其他模型文件，Resize节点名称可能会不一样，请根据实际情况修改
```
newnode10 = onnx.helper.make_node(
    'Resize',
    name = 'Resize_317',
    inputs = ['788', '792', '1061',],
    outputs = ['793'],
    coordinate_transformation_mode = 'asymmetric',
    cubic_coeff_a=-0.75,
    mode='nearest',
    nearest_mode='floor')

model.graph.node.remove(model.graph.node[7])
model.graph.node.insert(7,newnode)

model.graph.node.remove(model.graph.node[67])
model.graph.node.insert(67,newnode2)

model.graph.node.remove(model.graph.node[212])
model.graph.node.insert(212,newnode3)

model.graph.node.remove(model.graph.node[229])
model.graph.node.insert(229,newnode4)

model.graph.node.remove(model.graph.node[246])
model.graph.node.insert(246,newnode5)

model.graph.node.remove(model.graph.node[263])
model.graph.node.insert(263,newnode6)

model.graph.node.remove(model.graph.node[283])
model.graph.node.insert(283,newnode7)

model.graph.node.remove(model.graph.node[307])
model.graph.node.insert(307,newnode8)

model.graph.node.remove(model.graph.node[314])
model.graph.node.insert(314,newnode9)

model.graph.node.remove(model.graph.node[317])
model.graph.node.insert(317,newnode10)
```
修改后执行指令python3.7 resize.py ICNet.onnx  
新生成的onnx文件，通过Netron工具可以看到Resize节点的mode变为nearest
```
NODE PROPERTIES
	type     Resize
	name     Resize_67
ATTRINUTES
	coordinate_transformation_mode  asymmetric
	cubic_coeff_a  -0.75
	mode           nearest
	nearest_mode   floor
```

用修改后的onnx重新生成om文件，再次测试纯推理

纯推理测试结果：
```
bs1:
[INFO] Dataset number: 19 finished cost 172.557ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_ICNet_bs1_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 5.84567samples/s, ave_latency: 171.368ms
----------------------------------------------------------------


bs4:
[INFO] Dataset number: 19 finished cost 556.613ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_ICNet_bs4_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 7.18703samples/s, ave_latency: 139.277ms
----------------------------------------------------------------
```

用生成的om文件做精度后处理，测得bs1推理精度为64.1，未优化前推理精度为68.1，精度下降4, 百分比为5.8%。
```
Sample: 500, validation pixAcc: 93.158, mIoU: 66.633, time: 21.640s
Evaluate: Average mIoU: 0.641, Average pixAcc: 0.926, Average time: 21.761
```

#### 7.3.4 将PadV3D进行算子融合  

在cann 5.0.2版本后PadV3D算子已经可以和AvgPoolV2算子融合，缩短了运行时间，提高了性能。由于在蓝区测试的版本CANN 5.0.2.alpha003中，已经实现了PadV3D算子融合，因此测试过程默认已经优化。  

#### 7.3.5 优化FrameworkOP
使用CANN 5.0.1版本测试时，在onnx转om过程会报错，提示
```
not support ksize[2] 12 * ksize[3] *ksize 22 > 255 or strides[2] 10 > 63 strides[3] 21 > 63
```
在蓝区用CANN 5.0.2.alpha003版本测试，此问题已经规避（报错是因为模型走到aicore, 但是走到aicore需要对应conv2d模板支持对应场景，因为芯片限制,走不到aicore，只能走cpu, 导致算子性能低），不再报错，om转换过程也有warning提示，这种规避方法本身可以正常工作，但是会影响性能
```
ATC run success, welcome to the next use.
W11001: High-priority service of op[PartitionedCall_AvgPool_45_2] is invalid, low-priority service is used. It can work normally but may affect performance.
W11001: High-priority service of op[PartitionedCall_AvgPool_52_6] is invalid, low-priority service is used. It can work normally but may affect performance.
```
由于受芯片限制，该方法暂时不能优化。

#### 7.3.6 总结
优化方案共包括五种：  
（1）优化TransData，修改five_2_four.py  
（2）输出节点由float32改为float16  
（3）模型中Resize节点的mode由双线性为最近邻  
（4）将PadV3D进行算子融合  
（5）优化FrameworkOP框架  
由于在蓝区测试的版本CANN 5.0.2.alpha003中，已经实现了PadV3D算子融合，因此测试过程默认已经优化。同时方案（5）暂时无法实现，因此也无法比对性能。

主要对比前三种优化方案对推理性能的提升，如下：

| optimization point     | bs1(FPS)          | bs4(FPS)          | 
| :-------------------:  | :---------------: | :-------------:   | 
| initial                | 1.22*4=4.88       | 1.22*4=4.88       |  
| five_2_four.py         | 1.87*4=7.48       | 2.72*4=10.88      | 
| output_node: float16   | 3.67*4=14.68      | 3.71*4=14.84      |
| Resize: nearest        | 5.84*4=23.36      | 7.18*4=28.72      |

由以上数据可以看出：  
(1) bs1的吞吐率由最初的1.22提升至5.84，最终性能为5.84x4=23.36FPS  
(2) bs4的吞吐率由最初的1.22提升至7.18，最终性能为7.18x4=28.72FPS   

结论：
- 因为关键算子性能差，性能暂时无法达标。
- 最终精度测试，测得bs1推理精度为64.1，未优化前推理精度为68.1，精度下降5.8%，以上三种优化方案，除了第一种方案，其他方案虽然提升了性能，但是会使精度下降。