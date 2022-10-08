# PAMTRI Onnx模型端到端推理指导

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
	-   [6.1 离线推理Acc精度统计](#61-离线推理Acc精度统计)
	-   [6.2 精度对比](#62-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[[Tang, Zheng, et al. "Pamtri: Pose-aware multi-task learning for vehicle re-identification using highly randomized synthetic data." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.](https://arxiv.org/abs/2005.00673)](https://arxiv.org/abs/1712.00726)  

### 1.2 代码地址
[url=https://github.com/NVlabs/PAMTRI](https://github.com/NVlabs/PAMTRI)  
branch:master  
commit_id:a835c8cedce4ada1bc9580754245183d9f4aaa17

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.1.RC1
pytorch == 1.8.1
torchvision == 0.9.1
onnx == 1.8.0
```

### 2.2 python第三方库

```
Cython==0.29.25
h5py==3.7.0
numpy==1.21.4
Pillow==8.4.0
scipy==1.2.0
opencv-python==4.5.4.60
matplotlib>=2.1.0
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1. 准备pth权重文件  
使用训练好的pkl权重文件：densenet121-a639ec97.pth

下载路径：
https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/PAMTRI/densenet121-a639ec97.pth

2. 下载PAMTRI源码并安装

```shell
git clone https://github.com/NVlabs/PAMTRI # 克隆仓库的代码
cd PAMTRI/MultiTaskNet
git reset 25564bbebd3ccf11d853a345522e2d8c221b275d --hard
```

 **说明：**  
> 安装所需的依赖说明请参考PAMTRI/requirements.txt
>

3. PAMTRI代码迁移
通过打补丁的方式修改densetnet.py：
```shell
cd ./torchreid/models/
patch -p1 < densenet.patch
cd ..
```

4. 准备VeRi验证集，数据集获取参见本文第四章第一节 

5.  运行如下命令，生成PAMTRI_bs1.onnx(以batchsize1为例)

   运行“PAMTRI_pth2onnx.py”脚本：
```
python3.7 PAMTRI_pth2onnx.py -d veri -a densenet121 --root /opt/npu --load-weights models/densenet121-xent-htri-veri-multitask/model_best.pth.tar --output_path ./PAMTRI_bs1.onnx --multitask --batch_size 1
```

   --d：数据集名称，默认veri。

   --root：数据集路径，默认data。``

   --load-weights：pth权重文件读取路径。

   --output_path：onnx模型的输出路径。

   --batch_size：onnx模型的输入batch_size默认1。

   获得“PAMTRI_bs1.onnx”文件。

### 3.2 onnx转om模型

1. 设置环境变量
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 使用atc将onnx模型
${chip_name}可通过npu-smi info指令查看，例：310P3
![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

执行ATC命令：
```shell
atc --framework=5 \
--model=PAMTRI_bs1.onnx \
--output=PAMTRI_bs1 \
--input_format=NCHW \
--input_shape="input.1:1,3,256,256" \
--log=debug \
--soc_version=Ascend${chip_name} \
```

参数说明：
--model：为ONNX模型文件。
--framework：5代表ONNX模型。
--output：输出的OM模型。
--input_format：输入数据的格式。
--input_shape：输入数据的shape。
--log：日志级别。
--soc_version：处理器型号。

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
本模型支持Veri验证集。将获取到的VeRi数据集内容解压至“MultiTaskNet/veri”文件夹内。格式如下：
```
├─ data //数据集目录
│   ├─veri //veri数据集
│        ├─ image_train //VeRi数据集训练数据，本推理不需使用
│             ├─image1、2、3、4…
│        ├─ image_que    ry //VeRi数据集验证数据
│             ├─image1、2、3、4…
│        ├─ image_test //VeRi数据集测试数据
│             ├─image1、2、3、4…
│        ├─ label_train.csv //VeRi数据集训练数据的标签数据
│        ├─ label_query.csv //VeRi数据集验证数据的标签数据
│        ├─ label_test.csv //VeRi数据集测试数据的标签数据
```

### 4.2 数据集预处理
将原始数据集转换为模型输入的二进制数据。

执行PAMTRI_preprocess.py脚本。

```shell
cd  data/veri
mkdir heatmap_train
mkdir heatmap_query
mkdir heatmap_test
mkdir segment_train
mkdir segment_query
mkdir segment_test
cd -
python3.7 PAMTRI_preprocess.py  #不填参数，默认数值为在脚本文件中的设置
```
--d：原始数据验证集（.jpg）所在路径。

--save_path1：输出query的二进制文件（.bin）所在路径1。

--save_path2：输出gallery的二进制文件（.bin）所在路径2。

--query_dir：query（验证集）数据集的路径。

--gallery_dir：gallery（测试集）数据集的路径。

每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成“prep_dataset_query”、"prep_dataset_gallery"两个二进制文件夹。

### 4.3 生成数据集信息文件
使用benchmark推理需要输入二进制数据集的info文件，用于获取数据集。使用“gen_dataset_info.py”脚本，输入已经得到的二进制文件，输出生成二进制数据集的info文件。

生成BIN文件输入info文件

```shell
python3.7 gen_dataset_info.py bin ./prep_dataset_query ./prep_query_bin.info 256 256
python3.7 gen_dataset_info.py bin ./prep_dataset_gallery ./prep_gallery_bin.info 256 256
```
“bin”：生成的数据集文件格式。

“./prep_dataset_query”：预处理后的数据文件的**相对路径**。

“./prep_query_bin.info”：生成的数据集文件保存的路径。

“./prep_dataset_gallery”：预处理后的数据文件的**相对路径**。

“./prep_gallery_bin.info”：生成的数据集文件保存的路径。

“256”：图片宽高。

运行成功后，在当前目录中生成“prep_query_bin.info”和“prep_gallery_bin.info”。

## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)
### 5.2 离线推理
1.设置环境变量
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理
增加benchmark.{arch}可执行权限
```shell
chmod u+x benchmark.x86_64
```
执行推理
```shell
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./PAMTRI_bs1.om -input_text_path=./prep_query_bin.info -input_width=256 -input_height=256 -output_binary=True -useDvpp=False
                    
mv result/dumpOutput_device0 result/dumpOutput_device0_bs1_query
 
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./PAMTRI_bs1.om -input_text_path=./prep_gallery_bin.info -input_width=256 -input_height=256 -output_binary=True -useDvpp=False
 
mv result/dumpOutput_device0 result/dumpOutput_device0_bs1_gallery
```
参数说明：

- -model_type：模型类型。
- -om_path：om文件路径。
- -device_id：NPU设备编号。
- -batch_size：参数规模。
- -input_text_path：图片二进制信息。
- -input_width：输入图片宽度。
- -input_height：输入图片高度。
- -useDvpp：是否使用Dvpp。
- -output_binary：输出二进制形式。

推理后的输出默认在当前目录“result”下，“dumpOutput_device0_bs1_query”和“dumpOutput_device0_bs1_gallery”为处理后的数据集信息。


## 6 精度对比

-   **[离线推理mAP精度](#61-离线推理mAP精度)**  
-   **[开源mAP精度](#62-开源mAP精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理Acc精度统计

后处理统计Acc精度

调用“PAMTRI_postprocess.py”评测mAP精度
```python
python3.7 PAMTRI_postprocess.py --queryfeature_path=./result/dumpOutput_device0_bs1_query --galleryfeature_path=./result/dumpOutput_device0_bs1_gallery > result_bs1.json
```
参数说明：

--queryfeature_path：生成推理结果所在路径。

--galleryfeature_path：生成推理结果所在路径。

“result_bs1.json”：生成结果文件。

执行完后得到310P上的精度：
```
｜batchsize｜  mAP   ｜
｜    1    ｜ 68.64% ｜
｜    4    ｜ 68.64% ｜
｜    8    ｜ 68.64% ｜
｜    16   ｜ 68.64% ｜
｜    32   ｜ 68.64% ｜
｜    64   ｜ 68.64% ｜
```

### 6.2 精度对比
将得到的om离线模型推理Acc精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。

 ** 精度调试**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
离线推理的Interface throughputRate即为吞吐量，对于310，需要乘以4，310P只有一颗芯片，FPS为该值本身。

|        | 310      | 310P    | T4      | 310P/310    | 310P/T4     |
| ------ | -------- | ------- | ------- | ----------- | ----------- |
| bs1    | 728.732  | 878.002 | 127.816 | 1.204835248 | 6.869265194 |
| bs4    | 1067.644 | 1634.52 | 245.299 | 1.530959758 | 6.663378163 |
| bs8    | 1053.16  | 1420.6  | 288.224 | 1.348892856 | 4.928805374 |
| bs16   | 997.684  | 1589.93 | 314.009 | 1.593620826 | 5.06332621  |
| bs32   | 962.064  | 998.719 | 326.974 | 1.038100376 | 3.054429404 |
| bs64   | 942.732  | 951.187 | 331.704 | 1.008968615 | 2.86757772  |
| 最优bs | 1067.644 | 1634.52 | 331.704 | 1.530959758 | 4.927646335 |

最优batch：310P大于310的1.2；310P大于T4的1.6，性能达标

**性能优化：**  

>没有遇到性能不达标的问题，故不需要进行性能优化