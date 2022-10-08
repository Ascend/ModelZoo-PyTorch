# AlexNet Onnx模型端到端推理指导
- [1. 模型概述](#1)
    - [论文地址](#11)
    - [代码地址](#12)
- [2. 环境说明](#2)
    - [深度学习框架](#21)
    - [python第三方库](#22)
- [3. 模型转换](#3)
    - [pth转onnx模型](#31)
- [4. 数据预处理](#4)
    - [数据集获取](#41)
    - [数据集预处理](#42)
    - [生成数据集信息文件](#43)
- [5. 离线推理](#5)
    - [benchmark工具概述](#51)
    - [离线推理](#52)
- [6. 精度对比](#6)
    - [离线推理TopN精度](#61)
    - [开源TopN精度](#62)
    - [精度对比](#63)
- [7. 性能对比](#7)
    - [npu性能数据](#71)

## <a name="1">1. 模型概述</a>
- [论文地址](#11)
- [代码地址](#12)
### <a name="11">1.1 论文地址</a>
[AlexNet论文](https://wmathor.com/usr/uploads/2019/05/3327542327.pdf)
### <a name="12">1.2 代码地址</a>
[AlexNet代码](https://github.com/pytorch/examples/tree/master/imagenet)
> branch: master

> commit id: 49e1a8847c8c4d8d3c576479cb2fe2fd2ac583de
## <a name="2">2. 环境说明</a>
- [深度学习框架](#21)
- [python第三方库](#22)
### <a name="21">2.1 深度学习框架</a>

```
CANN 5.0.1
torch==1.8.1
torchvision==0.9.1
onnx==1.7.0
```
### <a name="22">2.2 python第三方库</a>

```
opencv-python==4.2.0.34
numpy==1.18.5
Pillow==7.2.0
```
 **说明：**
> X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
> Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装
## <a name="3">3. 模型转换</a>
- [pth转onnx模型](#31)
- [onnx转om模型](#32)
### <a name="31">3.1 pth转onnx模型</a>
1. 下载pth权重文件
[AlexNet预训练pth权重文件](https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth)

```
wget https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
```

> 该pth文件的md5sum值为：aed0662f397a0507305ac94ea5519309
2. AlexNet模型代码在torchvision里，需要安装torchvision
3. 编写pth2onnx脚本，生成onnx文件
>  **说明**
> 注意目前ATC支持的onnx算子版本为11
4. 执行pth2onnx.py脚本，生成onnx模型文件 
```
python3.7 pth2onnx.py alexnet-owt-4df8aa71.pth alexnet.onnx
```
### <a name="32">3.2 onnx转om模型</a>
1. 设置环境变量，请以实际安装环境配置环境变量。
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh

```

2. 使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)

```
atc --model=./alexnet.onnx --framework=5 --output=./onnx_alexnet_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,224,224" --log=debug --soc_version=Ascend310
```
若生成batch size为16的om模型，对应的命令为：

```
atc --model=./alexnet.onnx --framework=5 --output=./onnx_alexnet_bs16 --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --log=debug --soc_version=Ascend310
```
batch size为4、8、32的同上
## <a name="4">4. 数据预处理</a>
- [数据集获取](#41)
- [数据集预处理](#42)
- [生成数据集信息文件](#43)
### <a name="41">4.1 数据集获取</a>
该模型使用[ImageNet官网](http://www.image-net.org/)的5万张验证集进行测试，图片与标签分别存放在/opt/npu/imagenet/val与/opt/npu/imagenet/val_label.txt。
### <a name="42">4.2 数据集预处理</a>
1. 编写预处理脚本imagenet_torch_preprocess.py
预处理方式有两种：不使用aipp的二进制输入，以及使用aipp的jpg输入。这里使用第一种，需要先用脚本仿照github官网训练预处理方法处理数据，以获得最佳精度；
2. 执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 imagenet_torch_preprocess.py /opt/npu/imagenet/val ./pre_dataset
```
### <a name="43">4.3 生成数据集信息文件</a>
1. 编写生成数据集信息文件脚本get_info.py
2. 执行生成数据集信息脚本，生成数据集信息文件
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息

```
python3.7 get_info.py bin ./pre_dataset/ ./imagenet_prep_bin.info 224 224
```
## <a name="5">5. 离线推理</a>
- [benchmark工具概述](#51)
- [离线推理](#52)
### <a name="51">5.1 benchmark工具概述</a>
benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/ResNext50#https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)
将获取的工具包并解压，将benchmark工具放在当前目录下
### <a name="52">5.2 离线推理</a>
1. 设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2. 执行离线推理
运行如下命令进行离线推理：

```
./benchmark.x86_64 -model_type=vision -batch_size=1 -device_id=0 -om_path=./onnx_alexnet_bs1.om -input_text_path=./imagenet_prep_bin.info -input_width=224 -input_height=224 -useDvpp=false -output_binary=false
```
输出结果默认保存在当前目录/result/dumpOutput_device0中，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个txt文件。
## <a name="6">6. 精度对比</a>
- [离线推理TopN精度](#61)
- [开源TopN精度](#62)
- [精度对比](#63)
### <a name="61">6.1 离线推理TopN精度</a>
后处理与精度统计

调用vision_metric_ImageNet.py脚本与label比对，可以获得Accuracy Top5数据，结果保存在result/result.json中。

```
python3.7 vision_metric.py --benchmark_out ./result/dumpOutput_device0/ --anno_file /opt/npu/imagenet/val_label.txt --result_file ./result/result.json
```
查看输出的结果：

```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "56.53%"}, {"key": "Top2 accuracy", "value": "68.23%"}, {"key": "Top3 accuracy", "value": "73.49%"}, {"key": "Top4 accuracy", "value": "76.79%"}, {"key": "Top5 accuracy", "value": "79.08%"}]}
```

经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上
### <a name="62">6.2 开源TopN精度</a>
GPU上对torchvision里提供的pth文件进行推理，参考连接：[推理pth](https://github.com/pytorch/examples/tree/master/imagenet)
得到的结果是：
```
Acc@1 56.527
Acc@5 79.068
```
### <a name="63">6.3 精度对比</a>
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，如下表所示，精度下降在1%范围之内，故精度达标。
| 模型 | Acc@1 | Acc@5 |
|-------|-------|-------|
|pth模型推理结果（官方）|     56.527  |    79.068   |
| om模型离线推理结果|   56.530    |     79.080  |

 **说明：**
> 没有遇到精度不达标的问题，故不需要进行精度调试
## <a name="7">7. 性能对比</a>
- [npu性能数据](#71)
### <a name="71">7.1 npu性能数据</a>
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。
 **benchmark工具在整个数据集上推理获得性能数据:** 
1. batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：

```
[e2e] throughputRate: 211.301, latency: 236629
[data read] throughputRate: 225.214, moduleLatency: 4.44023
[preprocess] throughputRate: 224.859, moduleLatency: 4.44723
[infer] throughputRate: 212.679, Interface throughputRate: 317.707, moduleLatency: 4.03848
[post] throughputRate: 212.678, moduleLatency: 4.70194
```
Interface throughputRate: 317.707，317.707x4=1270.828fps。即是batch1 310单卡吞吐率

2. batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_0.txt。

得到的结果为：

```
[e2e] throughputRate: 312.638, latency: 159929
[data read] throughputRate: 321.245, moduleLatency: 3.11289
[preprocess] throughputRate: 319.722, moduleLatency: 3.12772
[infer] throughputRate: 314.855, Interface throughputRate: 1891.73, moduleLatency: 2.01078
[post] throughputRate: 19.6777, moduleLatency: 50.8189
```
Interface throughputRate: 1891.73，1891.73x4=7566.92fps。即是batch16 310单卡吞吐率
> 为了避免长期占用device， bs4,8,32使用纯推理测性能，其中，对bs4进行纯推理输入命令如下所示，其中batch_size=4表示bs的值，在对不同bs值对应的om模型进行推理时需要做出相应的更改：
> `./benchmark.x86_64 -device_id=0 -om_path=./onnx_alexnet_bs4.om -round=30 -batch_size=4`
> 推理结果保存在/result/PureInfer_perf_of_onnx_alexnet_bs4_in_device_0.txt中

3. 测试batch4的性能：

```
ave_throughputRate = 974.27samples/s, ave_latency = 1.02917ms

```
ave_throughputRate = 974.27, 974.27x4=3897.08fps。即是batch4 310单卡吞吐率

4. 测试batch8的性能：

```
ave_throughputRate = 1435.99samples/s, ave_latency = 0.697617ms
```
ave_throughputRate = 1435.99, 1435.99x4=5743.96fps。即是batch8 310单卡吞吐率

5. 测试batch32的性能

```
ave_throughputRate = 2186.17samples/s, ave_latency = 0.457931ms
```
ave_throughputRate = 2186.17, 2186.17x4=8744.68fps。即是batch32 310单卡吞吐率
 
**性能优化** 
> 从profiling看出MatMulV2耗时大，影响了网络性能，故不需要进行性能优化
