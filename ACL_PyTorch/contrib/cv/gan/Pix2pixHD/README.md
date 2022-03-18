# Pix2pixHD Onnx模型端到端推理指导
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
	-   [6.1 离线推理TopN精度统计](#61-离线推理TopN精度统计)
	-   [6.2 开源TopN精度](#62-开源TopN精度)
	-   [6.3 精度对比](#63-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)
	-   [7.2 T4性能数据](#72-T4性能数据)
	-   [7.3 性能对比](#73-性能对比)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[Pix2pixHD论文](https://arxiv.org/pdf/1711.11585.pdf)  

### 1.2 代码地址
[Pix2pixHD代码](https://github.com/NVIDIA/pix2pixHD)  
branch:master  
commit_id:5a2c87201c5957e2bf51d79b8acddb9cc1920b26  
备注：commit_id是指基于该次提交时的模型代码做推理，通常选择稳定版本的最后一次提交，或代码仓最新的一次提交  

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.0.1
pytorch >= 1.5.0
torchvision >= 0.6.0
onnx >= 1.9.0
```

### 2.2 python第三方库

```
numpy == 1.21.2
Pillow == 8.4.0
opencv-python == 4.5.2.54
dominate == 2.6.0
scipy == 1.7.2
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.Pix2pixHD模型代码下载
```
git clone https://github.com/NVIDIA/pix2pixHD.git
```
2.进入Pix2pixHD模型目录，利用diff文件对原开源项目进行细微改动。
```
cd pix2pixHD
patch -p1 < ../pix2pixhd_npu.diff
cd ..
```
3.Pix2pixHD模型是基于GAN网络的生成模型，官方在github项目上发布了预训练好的生成器模型latest_net_G.pth，下载该模型到本目录。进入pix2pixHD目录，创建./checkpoints/label2city_1024p/目录，并将latest_net_G.pth放置在该目录下。

```
cd pix2pixHD
mkdir ./checkpoints/label2city_1024p/
mv ../latest_net_G.pth ./checkpoints/label2city_1024p/
cd ..
```

4.编写pth2onnx脚本pix2pixhd_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

5.执行pth2onnx脚本，生成onnx模型文件
```
python pix2pixhd_pth2onnx.py --load_pretrain ./pix2pixHD/checkpoints/label2city_1024p --output_file pix2pixhd.onnx
```

 **模型转换要点：**  
>模型转换时需要下载预训练模型并将其放置在项目合适位置。

### 3.2 onnx转om模型

1.设置环境变量
```
source env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)
```
atc --framework=5 --model=./pix2pixhd.onnx --input_format=NCHW --input_shape="input_concat:1,36,1024,2048" --output=pix2pixhd_bs1 --log=debug --soc_version=Ascend310
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型采用[cityscapes数据集](https://www.cityscapes-dataset.com/downloads/)的数据集，打开cityscapes官方网址，注册教育邮箱并登陆，进入下载界面，下载gtFine_trainvaltest.zip和leftImg8bit_trainvaltest.zip两个压缩包到本目录。
**数据集获取注意：**
1.cityscapes数据集仅对学术研究开放，因此在官网下载该数据即注册邮箱需为含有edu关键字的邮箱才可以通过验证。
2.Pix2pixHD模型较为复杂，生成的onnx模型与om模性会占用较大空间；并且数据集在预处理之后每一张输入图片会生成一个大小约为300M的bin文件，因此本模型的推理仅在bs=1的情况下进行，bs高于1推理机器会因为内存不够而报错。同时考虑到每一张输入图片在预处理后占用空间较大，因此本推理仅采用10张测试集图片进行测试精度和性能，而Pix2pixHD官方仓库中自带10张测试集图片，因此就完成本推理任务而言，数据集可以不进行下载。
3.本模型由于硬件限制，只在Pix2pixHD仓库中自带的10张测试机图片进行推理，因此若只是完成本推理任务，数据集不必要下载，具体原因见上一条。若想要在整个数据集的测试集上推理，则需要按照以上步骤下载数据集。
4.(本次推理不需要执行以下命令，本条只是用于在全部测试集推理时执行)倘若在整个测试集推理本模型，需将gtFine_trainvaltest.zip和leftImg8bit_trainvaltest.zip进行解压，并且进入pix2pixHD目录并且删除datasets子目录。执行deal.py脚本将数据集从转换为pix2pixHD项目所需要的组织格式并且放置在新建的pix2pixHD/datasets目录下。
```
unzip gtFine_trainvaltest.zip
unzip leftImg8bit_trainvaltest
cd pix2pixHD
rm -rf datasets
cd ..
python datasets_deal.py ./cityscapes/test_inst ./cityscapes/test_label ./gtFine/test
cp -r ./cityscapes ./pix2pixHD/datasets
```

### 4.2 数据集预处理
1.执行预处理脚本，生成数据集预处理后的bin文件
```
python pix2pixhd_preprocess.py ./pix2pixHD/datasets/cityscapes ./prep_dataset
```

### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python get_info.py bin ./prep_dataset ./pix2pixhd_prep_bin.info 2048 1024
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息  
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)
### 5.2 离线推理
1.设置环境变量
```
source env.sh
```
2.执行离线推理
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=pix2pixhd_bs1.om -input_text_path=./pix2pixhd_prep_bin.info -input_width=2048 -input_height=1024 -output_binary=True -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_deviceX(X为对应的device_id)，每个输入对应一个_X.bin文件的输出。

3.将输出bin文件解析为jpg文件，并保存在./generated目录下
```
python pix2pixhd_postprocess.py ./result/dumpOutput_device0 ./generated
```

## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度

Pix2pixHD模型是生成模型，原论文中作者对于该模型的评估方式是重新训练了一个评估模型pspnet,并用该评估模型来评估Pix2pixHD模型生成模型的质量，但是由于作者并没有给出pspnet的相关代码链接以及权重，因此本次精度对比采用npu推理与gpu生成结果对比的方式来进行。具体的操作方式为给gpu模型与om模型相同的输入，人为观察输出图片的质量，两者相近，则认为达标。
因为在后续性能对比中，gpu仍需要采用在线推理的方式来进行，为了保持一致性，将Pix2pixHD模型部署到T4类型的GPU，仅需在T4上跑一次即可拿到精度与性能数据。

Pix2pixHD模型的离线推理生成的图片保存在目录下。

### 6.2 GPU推理精度
1.在装有T4GPU的服务器上下载Pix2pixHD模型，并将本目录下的pix2pixhd_gpu.diff文件传至T4服务器Pix2pixHD同一目录下,以下步骤均为在T4服务器上操作。
```
git clone https://github.com/NVIDIA/pix2pixHD.git

```
2.进入Pix2pixHD模型目录，利用diff文件对原开源项目进行细微改动。
```
cd pix2pixHD
patch -p1 < ../pix2pixhd_gpu.diff
cd ..
```
3.Pix2pixHD模型是基于GAN网络的生成模型，官方在github项目上发布了预训练好的生成器模型latest_net_G.pth，下载该模型到本目录。进入pix2pixHD目录，创建./checkpoints/label2city_1024p/目录，并将latest_net_G.pth放置在该目录下。

```
cd pix2pixHD
mkdir ./checkpoints/label2city_1024p/
mv ../latest_net_G.pth ./checkpoints/label2city_1024p/
cd ..
```

3.进入Pix2pixHD目录执行bash ./scripts/test_1024p.sh命令，此时将在Pix2pix/pix2pixHD/results/label2city_1024p/test_latest/images目录下生成图片,同时会在终端输出日志信息。
```
cd pix2pixHD
bash ./scripts/test_1024p.sh
cd ..
```
### 6.3 精度对比
将om离线推理审查各行的图片与gpu离线推理生成的图片人为进行观察，10张图片完全一致，精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  
-   **[T4性能数据](#72-T4性能数据)**  
-   **[性能对比](#73-性能对比)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。 
但由于Pix2pixHD模型模型较大，同时输入图片转换得到的bin文件也较大，由于硬件的限制的原因，本模型推理只测试bs=1的情形，bs=4或者更大时310推理卡内存不足，故不进行推理测试。同时由于一张输入图片转换而成的bin文件大约为300M,所占空间较大，因此本次推理只在Pix2pixHD仓库中自带的10张测试集图片进行推理测试，而不在整个测试集上测试(整个测试集共1525张图片，每张图片预处理后转换而成的bin文件为300M，整个测试集全部测试服务器硬盘容量不足)。
1.benchmark工具推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  
```
[e2e] throughputRate: 0.31695, latency: 47326.1
[data read] throughputRate: 4.29009, moduleLatency: 233.095
[preprocess] throughputRate: 0.987048, moduleLatency: 1013.12
[infer] throughputRate: 0.329787, Interface throughputRate: 0.422335, moduleLatency: 2985.64
[post] throughputRate: 0.329426, moduleLatency: 3035.58
```
Interface throughputRate: 0.422335，0.422335x4=1.68934既是batch1 310单卡吞吐率  
  
### 7.2 T4性能数据
在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2  

Pix2pixHD模型中采用了reflection的pad方式，由于TensorRT目前尚不支持该pad方式，因此Pix2pixHD模型无法采用离线推理，故采用在线推理方式，在6.2中GPU推理精度中在生成图片的同时终端也会输出bs=1的gpu的性能数据，对该输出数据进行简单处理得到
batch1性能：
```
本次生成图像耗时3.2349298000335693s
process image... ['./datasets/cityscapes/test_label/frankfurt_000000_000576_gtFine_labelIds.png']
本次生成图像耗时0.49680328369140625s
process image... ['./datasets/cityscapes/test_label/frankfurt_000000_001236_gtFine_labelIds.png']
本次生成图像耗时0.49724769592285156s
process image... ['./datasets/cityscapes/test_label/frankfurt_000000_003357_gtFine_labelIds.png']
本次生成图像耗时0.5650396347045898s
process image... ['./datasets/cityscapes/test_label/frankfurt_000000_011810_gtFine_labelIds.png']
本次生成图像耗时0.5557215213775635s
process image... ['./datasets/cityscapes/test_label/frankfurt_000000_012868_gtFine_labelIds.png']
本次生成图像耗时0.5854263305664062s
process image... ['./datasets/cityscapes/test_label/frankfurt_000001_013710_gtFine_labelIds.png']
本次生成图像耗时0.4999239444732666s
process image... ['./datasets/cityscapes/test_label/frankfurt_000001_015328_gtFine_labelIds.png']
本次生成图像耗时0.5007188320159912s
process image... ['./datasets/cityscapes/test_label/frankfurt_000001_023769_gtFine_labelIds.png']
本次生成图像耗时0.503685712814331s
process image... ['./datasets/cityscapes/test_label/frankfurt_000001_028335_gtFine_labelIds.png']
本次生成图像耗时0.5905485153198242s
process image... ['./datasets/cityscapes/test_label/frankfurt_000001_032711_gtFine_labelIds.png']
本次生成图像耗时0.5026726722717285s
process image... ['./datasets/cityscapes/test_label/frankfurt_000001_033655_gtFine_labelIds.png']
本次生成图像耗时0.5029397010803223s
process image... ['./datasets/cityscapes/test_label/frankfurt_000001_042733_gtFine_labelIds.png']
本次生成图像耗时0.5053918361663818s
process image... ['./datasets/cityscapes/test_label/frankfurt_000001_047552_gtFine_labelIds.png']
本次生成图像耗时0.5058104991912842s
process image... ['./datasets/cityscapes/test_label/frankfurt_000001_054640_gtFine_labelIds.png']
本次生成图像耗时0.5039389133453369s
process image... ['./datasets/cityscapes/test_label/frankfurt_000001_055387_gtFine_labelIds.png']

```
通过上述输出日志去掉第一次生成所需较长时间，剩余14次图像生成计算得到T4在线推理生成一张图片平均时间为0.5225s
  
### 7.3 性能对比
在T4上bs为1时的跑一次测试图像的时间平均大概是0.5225s
因此在t4上的性能是1000/(522.5/1)=1.9138
bs为1是，npu性能为0.422335x4=1.68934  
由于Pix2pixHD模型对于性能的要求是达到GPU性能的一半，因为1.68934>(1.9138/2),故本模型性能达标。
  
 **性能优化：**  
>没有遇到性能不达标的问题，故不需要进行性能优化

