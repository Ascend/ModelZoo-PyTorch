# Squeezenet1_1 Onnx模型端到端推理指导
-   [1 模型概述](#1-模型概述)  
	-   [1.1 论文地址](#11-论文地址)
	-   [1.2 代码地址](#12-代码地址)
-   [2 环境说明](#2-环境说明)
	-   [2.1 深度学习框架](#21-深度学习框架)
	-   [2.2 python第三方库](#22-python第三方库)
-   [3 数据集预处理](#3-数据集预处理)
	-   [3.1 数据集获取](#31-数据集获取)
	-   [3.2 数据集预处理](#32-数据集预处理)
	-   [3.3 生成数据集信息文件](#33-生成数据集信息文件)
-   [4 模型转换](#4-模型转换)
	-   [4.1 pth转onnx模型](#41-pth转onnx模型)
	-   [4.2 onnx转om模型](#42-onnx转om模型)
-   [5 离线推理](#5-离线推理)
	-   [5.1 benchmark工具概述](#51-benchmark工具概述)
	-   [5.2 离线推理](#52-离线推理)
-   [6 精度对比](#6-精度对比)
	-   [6.1 离线推理精度统计](#61-离线推理精度统计)
	-   [6.2 开源精度](#62-开源精度)


## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[Squeezenet1_1论文](https://arxiv.org/abs/1602.07360)  

### 1.2 代码地址
[Squeezenet1_1代码](https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py)  
branch:master  
commit id:d1f1a5445dcbbd0d733dc38a32d9ae153337daae  


## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.1.RC1

torch == 1.6.0
torchvision == 0.7.0
onnx == 1.7.0
```

### 2.2 python第三方库

```
numpy == 1.18.5
pillow == 7.2.0
opencv-python == 4.2.0.34
```


**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 数据集预处理

-   **[数据集获取](#31-数据集获取)**  

-   **[数据集预处理](#32-数据集预处理)**  

-   **[生成数据集信息文件](#33-生成数据集信息文件)** 

### 3.1 数据集获取
该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/root/datasets/imagenet/val与/root/datasets/imagenet/val_label.txt。

```
├── dataset
	├── imagenet
		├──val		
		├──val_label.txt
```


### 3.2 数据集预处理
1.预处理脚本imagenet_torch_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件

```
python3.7 imagenet_torch_preprocess.py squeezenet1_1 /home/HwHiAiUser/dataset/imagenet/val./prep_dataset
```

第一个参数为原始数据验证集（.jpeg）所在路径，第二个参数为输出的二进制文件（.bin）所在路径。每个图像对应生成一个二进制文件。

### 3.3 生成数据集信息文件

1.生成数据集信息文件脚本gen_dataset_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```
python3.7 gen_dataset_info.py bin ./prep_dataset ./squeezenet1_1_prep_bin.info 224 224
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息



## 4 模型转换

-   **[pth转onnx模型](#41-pth转onnx模型)**  

-   **[onnx转om模型](#42-onnx转om模型)**  

### 4.1 pth转onnx模型

1.下载pth权重文件  
[Squeezenet1_1预训练pth权重文件](https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth)  
文件md5sum: 46a44d32d2c5c07f7f66324bef4c7266  

```
wget https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth
```

2.squeezenet1_1模型代码在torchvision里，安装torchvision，arm下需源码安装，参考torchvision官网，若安装过程报错请百度解决

```
git clone https://github.com/pytorch/vision
cd vision
python3.7 setup.py install /
cd ..
```

3.编写pth2onnx脚本squeezenet1_1_pth2onnx.py  

 **说明：**  

>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件

```
python3.7 squeezenet1_1_pth2onnx.py squeezenet1_1-f364aa15.pth squeezenet1_1.onnx
```

 **模型转换要点：**  

>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明  

### 4.2 onnx转om模型

1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)

```
atc --framework=5 --model=squeezenet1_1.onnx --input_format=NCHW --input_shape="image:16,3,224,224"  --output=squeezenet1_1_bs16 --log=debug --soc_version=Ascend${chip_name} --enable_small_channel=1 --insert_op_conf=aipp.config
```

--${chip_name}可通过npu-smi info指令查看
![输入图片说明](../../../../images/310P3.png)

--model：为ONNX模型文件。

--framework：5代表ONNX模型。

--output：输出的OM模型。

--input_format：输入数据的格式。

--input_shape：输入数据的shape，第一个数字为batchsize。

--log：日志级别。

--soc_version：处理器型号。

--auto_tune_mode：auto_tune模式。

--insert_op_conf: 插入算子的配置文件路径与文件名，例如aipp预处理算子

## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)
### 5.2 离线推理
1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.增加执行权限

```
chmod u+x benchmark.x86_64
```

3.执行离线推理

以batchsize=16为例

```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=squeezenet1_1_bs16.om -input_text_path=./squeezenet1_1_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_device{0}，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度统计

后处理统计TopN精度

调用imagenet_acc_eval.py脚本推理结果与val_map比对，可以获得Accuracy Top5数据，结果保存在result.json中。
```
python3.7 imagenet_acc_eval.py result/dumpOutput_device0/ /home/HwHiAiUser/dataset/imagenet/val_label.txt ./ result.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。   
查看输出结果：

```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "58.19%"}, {"key": "Top2 accuracy", "value": "69.67%"}, {"key": "Top3 accuracy", "value": "75.07%"}, {"key": "Top4 accuracy", "value": "78.3%"}, {"key": "Top5 accuracy", "value": "80.61%"}]}
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上   

### 6.2 开源精度
[torchvision官网精度](https://pytorch.org/vision/stable/models.html)

```
Model               Acc@1     Acc@5
Squeezenet1_1       58.178    80.624
```



