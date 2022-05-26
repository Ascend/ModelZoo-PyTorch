# Wide_ResNet101_2 Onnx模型端到端推理指导

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
	-   [6.2 开源精度](#62-开源精度)
	-   [6.3 精度对比](#63-精度对比)
	
- [7 性能对比](#7-性能对比)

  -   [7.1 npu性能数据](#71-npu性能数据)
  -   [7.2 T4性能数据](#72-T4性能数据)
  -   [7.3 性能对比](#73-性能对比)

  



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址

[Wide_ResNet论文](https://arxiv.org/pdf/1605.07146.pdf)  

### 1.2 代码地址

[Wide_ResNet代码](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)

branch:master
commit id:7d955df73fe0e9b47f7d6c77c699324b256fc41f

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架

```
CANN 5.1RC1

pytorch == 1.11.0
torchvision = 0.6.0
onnx = 1.11.0
```

### 2.2 python第三方库

```
numpy == 1.21.2
Pillow == 8.3.2
opencv-python == 4.5.3.56
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1. 下载pth权重文件  

[wrn101_2权重文件下载](https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth)

文件md5sum:  5961435974bb43104b5a3180fea7c2c4 

```
wget https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth
```



2. 下载模型代码

```
git clone https://github.com/pytorch/vision
cd vision
git reset 7d955df73fe0e9b47f7d6c77c699324b256fc41f --hard
python3.7 setup.py install
cd ..
```

3. 编写pth2onnx脚本wrn101_2_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4. 执行pth2onnx脚本，生成onnx模型文件


```python
python3.7 wrn101_2_pth2onnx.py wide_resnet101_2-32ee1156.pth wrn101_2_pth.onnx
```

 **模型转换要点：**  
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明

### 3.2 onnx转om模型

1.设置环境变量

```python
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01

```shell
#310:
atc --framework=5 --model=wrn101_2_pth.onnx --output=wrn101_2_bs16 --input_format=NCHW --input_shape="image:16,3,224,224" --log=debug --soc_version=Ascend310

#310p:
atc --framework=5 --model=wrn101_2_pth.onnx --output=wrn101_2_bs16 --input_format=NCHW --input_shape="image:16,3,224,224" --log=debug --soc_version=Ascend710
```



## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/opt/npu/imagenet/val与/opt/npu/imagenet/val_label.txt

### 4.2 数据集预处理

1.预处理脚本imagenet_torch_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件

```
python3.7 imagenet_torch_preprocess.py resnet /opt/npu/imagenet/val ./prep_dataset
```
### 4.3 生成数据集信息文件

1.生成数据集信息文件脚本gen_dataset_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```python
python3.7 gen_dataset_info.py bin ./prep_dataset ./wrn101_2_prep_bin.info 224 224
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息



## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN V100R020C10 推理benchmark工具用户指南 01

### 5.2 离线推理

1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理

```python
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=wrn101_2_bs16.om -input_text_path=./wrn101_2_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_device{0}，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。



## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度统计

后处理统计TopN精度

调用imagenet_acc_eval.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。
```
python3.7 imagenet_acc_eval.py result/dumpOutput_device0/ /opt/npu/imagenet/val_label.txt ./ result.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  
查看输出结果：

```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "78.84%"}, {"key": "Top2 accuracy", "value": "88.41%"}, {"key": "Top3 accuracy", "value": "91.66%"}, {"key": "Top4 accuracy", "value": "93.26%"}, {"key": "Top5 accuracy", "value": "94.29%"}]}
```

### 6.2 开源精度

[torchvision官网精度](https://pytorch.org/vision/stable/models.html)

```
Model                 Acc@1       Acc@5
wide_resnet101_2      78.848     94.284
```
### 6.3 精度对比

将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，统计结果如下。精度下降在1%范围之内，故精度达标。  

```
                Acc@1      Acc@5
bs1             78.84      94.29
bs16            78.85      94.29
```

 **精度调试：**  

>没有遇到精度不达标的问题，故不需要进行精度调试



## 7 性能对比

-   **[310npu性能数据](#71-310npu性能数据)**  
-   **[310p npu性能数据](#72-310p npu性能数据)**  
-   **[T4性能数据](#73-T4性能数据)**  
-   **[性能对比](#73-性能对比)**  

### 7.1 310 npu性能数据

benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。

1.benchmark工具在整个数据集上推理获得性能数据

batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：

```
[e2e] throughputRate: 105.964, latency: 471859
[data read] throughputRate: 112.337, moduleLatency: 8.90179
[preprocess] throughputRate: 111.931, moduleLatency: 8.93404
[infer] throughputRate: 106.222, Interface throughputRate: 129.018, moduleLatency: 8.70911
[post] throughputRate: 106.222, moduleLatency: 9.41428
```

Interface throughputRate: 129.018，129.018x4=516.072即是batch1 310单卡吞吐率

batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：

```
[e2e] throughputRate: 119.367, latency: 418876
[data read] throughputRate: 126.399, moduleLatency: 7.91145
[preprocess] throughputRate: 126.137, moduleLatency: 7.92786
[infer] throughputRate: 119.647, Interface throughputRate: 170.965, moduleLatency: 7.27049
[post] throughputRate: 7.47771, moduleLatency: 133.731
```

Interface throughputRate: 170.965，170.965x4=683.86即是batch1 310单卡吞吐率

batch4性能：

```
[e2e] throughputRate: 101.852, latency: 490910
[data read] throughputRate: 107.479, moduleLatency: 9.30415
[preprocess] throughputRate: 107.138, moduleLatency: 9.33379
[infer] throughputRate: 102.151, Interface throughputRate: 157.248, moduleLatency: 8.78044
[post] throughputRate: 25.5375, moduleLatency: 39.1581
```

batch4 310单卡吞吐率：157.248x4=628.992fps
batch8性能：

```
[e2e] throughputRate: 106.178, latency: 470906
[data read] throughputRate: 112.19, moduleLatency: 8.91342
[preprocess] throughputRate: 111.912, moduleLatency: 8.93559
[infer] throughputRate: 106.421, Interface throughputRate: 157.434, moduleLatency: 8.37058
[post] throughputRate: 13.3024, moduleLatency: 75.1742
```

batch8 310单卡吞吐率：157.434x4=629.736fps
batch32性能：

```
[e2e] throughputRate: 102.387, latency: 488344
[data read] throughputRate: 108.61, moduleLatency: 9.20728
[preprocess] throughputRate: 108.389, moduleLatency: 9.22602
[infer] throughputRate: 102.81, Interface throughputRate: 139.595, moduleLatency: 8.59119
[post] throughputRate: 3.2138, moduleLatency: 311.159
```

batch32 310单卡吞吐率：139.595x4=591.376fps

batch64性能：

```
[e2e] throughputRate: 90.2739, latency: 553870
[data read] throughputRate: 95.3212, moduleLatency: 10.4908
[preprocess] throughputRate: 95.1398, moduleLatency: 10.5108
[infer] throughputRate: 90.4519, Interface throughputRate: 125.414, moduleLatency: 9.90415
[post] throughputRate: 1.41463, moduleLatency: 706.898
```

batch64 310单卡吞吐率：125.414x4=501.656fps



### 7.2 310p npu性能数据

1.benchmark工具在整个数据集上推理获得性能数据

batch1的性能（没有执行aoe），benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：

```
[e2e] throughputRate: 195.097, latency: 256282
[data read] throughputRate: 201.138, moduleLatency: 4.97171
[preprocess] throughputRate: 200.301, moduleLatency: 4.99249
[infer] throughputRate: 196.592, Interface throughputRate: 265.092, moduleLatency: 4.45654
[post] throughputRate: 196.592, moduleLatency: 5.08668
```

Interface throughputRate: 265.092是batch1 310p调优前的单卡吞吐率



batch1的性能（执行aoe），benchmark工具在整个数据集上推理后生成result_aoe/perf_vision_batchsize_1_device_0.txt：

```
aoe --framewor=5 --model=wrn101_2_pth.onnx --job_type=2 --output=wrn101_2_bs1_aoe --input_shape "image:1,3,224,224" --log=error
```

```
[e2e] throughputRate: 105.883, latency: 472219
[data read] throughputRate: 107.982, moduleLatency: 9.26082
[preprocess] throughputRate: 107.76, moduleLatency: 9.27989
[infer] throughputRate: 106.271, Interface throughputRate: 465.954, moduleLatency: 7.71249
[post] throughputRate: 106.27, moduleLatency: 9.40996
```

Interface throughputRate: 465.954是batch1 310p调优后的单卡吞吐率



batch8的性能（没有执行aoe），benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：

```
[e2e] throughputRate: 166.691, latency: 299956
[data read] throughputRate: 167.683, moduleLatency: 5.96365
[preprocess] throughputRate: 167.347, moduleLatency: 5.97561
[infer] throughputRate: 167.444, Interface throughputRate: 322.615, moduleLatency: 4.05673
[post] throughputRate: 20.93, moduleLatency: 47.7783
```

Interface throughputRate: 322.615是batch8 310p调优前的单卡吞吐率



batch8的性能（执行aoe），benchmark工具在整个数据集上推理后生成result_aoe/perf_vision_batchsize_8_device_0.txt：

```
aoe --framewor=5 --model=wrn101_2_pth.onnx --job_type=2 --output=wrn101_2_bs8_aoe --input_shape "image:8,3,224,224" --log=error
```

```
[e2e] throughputRate: 144.975, latency: 344886
[data read] throughputRate: 145.776, moduleLatency: 6.85985
[preprocess] throughputRate: 145.478, moduleLatency: 6.87388
[infer] throughputRate: 145.664, Interface throughputRate: 1003.8, moduleLatency: 2.76154
[post] throughputRate: 18.2077, moduleLatency: 54.922
```

Interface throughputRate: 1003.8是batch8 310p调优后的单卡吞吐率



batch4性能（没有执行aoe）：

```
[e2e] throughputRate: 179.241, latency: 278954
[data read] throughputRate: 180.657, moduleLatency: 5.53536
[preprocess] throughputRate: 180.266, moduleLatency: 5.54736
[infer] throughputRate: 180.403, Interface throughputRate: 318.947, moduleLatency: 4.10043
[post] throughputRate: 45.1001, moduleLatency: 22.1729
```

batch4 310p调优前的单卡吞吐率：318.947fps

batch4性能（执行aoe）：

```
aoe --framewor=5 --model=wrn101_2_pth.onnx --job_type=2 --output=wrn101_2_bs4_aoe --input_shape "image:4,3,224,224" --log=error
```

```
[e2e] throughputRate: 168.435, latency: 296851
[data read] throughputRate: 169.043, moduleLatency: 5.91564
[preprocess] throughputRate: 168.686, moduleLatency: 5.92816
[infer] throughputRate: 169.004, Interface throughputRate: 946.034, moduleLatency: 2.43772
[post] throughputRate: 42.2506, moduleLatency: 23.6683
```

batch4 310p调优后的单卡吞吐率：946.034fps



batch16性能（没有执行aoe）：

```
[e2e] throughputRate: 243.49, latency: 205347
[data read] throughputRate: 259.265, moduleLatency: 3.85705
[preprocess] throughputRate: 257.943, moduleLatency: 3.87683
[infer] throughputRate: 245.795, Interface throughputRate: 407.177, moduleLatency: 3.29788
[post] throughputRate: 15.3612, moduleLatency: 65.0991
```

batch16 310p调优前的单卡吞吐率：407.177fps



batch16性能（执行aoe）：

```
aoe --framewor=5 --model=wrn101_2_pth.onnx --job_type=2 --output=wrn101_2_bs16_aoe --input_shape "image:16,3,224,224" --log=error
```

```
[e2e] throughputRate: 116.841, latency: 427930
[data read] throughputRate: 117.63, moduleLatency: 8.50121
[preprocess] throughputRate: 117.471, moduleLatency: 8.51277
[infer] throughputRate: 117.184, Interface throughputRate: 1001.67, moduleLatency: 3.9011
[post] throughputRate: 7.32378, moduleLatency: 136.542
```

batch16 310p调优后的单卡吞吐率：1001.67fps



batch32性能（没有执行aoe）：

```
[e2e] throughputRate: 214.139, latency: 233494
[data read] throughputRate: 215.786, moduleLatency: 4.63422
[preprocess] throughputRate: 214.711, moduleLatency: 4.65742
[infer] throughputRate: 215.578, Interface throughputRate: 410.758, moduleLatency: 3.28237
[post] throughputRate: 6.7386, moduleLatency: 148.399
```

batch32 310p调优前的单卡吞吐率：410.758fps



batch32性能（执行aoe）：

```
aoe --framewor=5 --model=wrn101_2_pth.onnx --job_type=2 --output=wrn101_2_bs32_aoe --input_shape "image:32,3,224,224" --log=error
```

```
[e2e] throughputRate: 99.8225, latency: 500889
[data read] throughputRate: 102.585, moduleLatency: 9.74806
[preprocess] throughputRate: 102.428, moduleLatency: 9.76292
[infer] throughputRate: 100.129, Interface throughputRate: 895.306, moduleLatency: 4.0268
[post] throughputRate: 3.12995, moduleLatency: 319.494
```

batch32 310p调优后的单卡吞吐率：895.306fps



batch64性能（没有执行aoe）：

```
[e2e] throughputRate: 188.317, latency: 265509
[data read] throughputRate: 193.949, moduleLatency: 5.15601
[preprocess] throughputRate: 193.029, moduleLatency: 5.18057
[infer] throughputRate: 189.853, Interface throughputRate: 275.251, moduleLatency: 4.45045
[post] throughputRate: 2.96915, moduleLatency: 336.797
```

batch64 310p调优前的单卡吞吐率：275.251fps



batch64性能（执行aoe）：

```
aoe --framewor=5 --model=wrn101_2_pth.onnx --job_type=2 --output=wrn101_2_bs64_aoe --input_shape "image:64,3,224,224" --log=error
```

```
[e2e] throughputRate: 99.3163, latency: 503442
[data read] throughputRate: 99.8186, moduleLatency: 10.0182
[preprocess] throughputRate: 99.6759, moduleLatency: 10.0325
[infer] throughputRate: 99.7346, Interface throughputRate: 823.149, moduleLatency: 4.09318
[post] throughputRate: 1.55981, moduleLatency: 641.103
```

batch64 310p调优后的单卡吞吐率：823.149fps

### 7.3 T4性能数据

在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2

batch1性能：

```
trtexec --onnx=wrn101_2_pth.onnx --fp16 --shapes=image:1x3x224x224 --threads
```

batch1 t4单卡吞吐率：244.615fps

gpu T4是4个device并行执行的结果，mean是时延（tensorrt的时延是batch个数据的推理时间），即吞吐率的倒数乘以batch。其中--fp16是算子精度，目前算子精度只测--fp16的。注意--shapes是onnx的输入节点名与shape，当onnx输入节点的batch为-1时，可以用同一个onnx文件测不同batch的性能，否则用固定batch的onnx测不同batch的性能不准

batch64性能：

```
trtexec --onnx=wrn101_2_pth.onnx --fp16 --shapes=image:64x3x224x224 --threads
```

batch64 t4单卡吞吐率：572.999fps



batch4 t4单卡吞吐率：421.312fps

batch8 t4单卡吞吐率：482.41fps

batch16 t4单卡吞吐率：538.522fps

batch32 t4单卡吞吐率：570.786fps

### 7.4 性能对比

|                |   310   |  310p   | aoe后的310p |   T4    | 310p/310(aoe后) | 310p/T4(aoe后) |
| :------------: | :-----: | :-----: | :---------: | :-----: | :-------------: | :------------: |
|      bs1       | 516.072 | 265.092 |   465.954   | 244.615 |      0.903      |     1.905      |
|      bs4       | 628.992 | 318.947 |   946.034   | 421.312 |      1.504      |     2.245      |
|      bs8       | 629.736 | 322.615 |  1003.800   | 482.410 |      1.594      |     2.081      |
|      bs16      | 683.860 | 407.177 |  1001.670   | 538.522 |      1.465      |     1.860      |
|      bs32      | 591.376 | 410.758 |   895.306   | 570.786 |      1.785      |     1.569      |
|      bs64      | 501.656 | 275.251 |   823.149   | 572.999 |      1.641      |     1.436      |
| 最优batch_szie | 683.860 | 410.758 |  1003.800   | 572.999 |      1.468      |     1.752      |

最优batch下，已经达到：

1. 310p调优后的最优batch性能 >=1.2倍310最优batch性能; 

2. 310p的最优batch性能 >=1.6倍T4最优batch性能。

