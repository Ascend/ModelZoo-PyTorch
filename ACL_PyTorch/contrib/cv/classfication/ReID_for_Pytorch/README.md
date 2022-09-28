# ReID ONNX模型端到端推理指导
- [ReID ONNX模型端到端推理指导](#ReID-onnx模型端到端推理指导)
    - [1 模型概述](#1-模型概述)
        - [1.1 论文地址](#11-论文地址)
        - [1.2 代码地址](#12-代码地址)
    - [2 环境说明](#2-环境说明)
        - [2.1 深度学习框架](#21-深度学习框架)
        - [2.2 python第三方库](#22-python第三方库)
    - [3 模型转换](#3-模型转换)
        - [3.1 pth转onnx模型](#31-pth转onnx模型)
        - [3.2 onnx转om模型](#32-onnx转om模型)
    - [4 数据集预处理](#4-数据集预处理)
        - [4.1 数据集获取](#41-数据集获取)
        - [4.2 数据集预处理](#42-数据集预处理)
        - [4.3 生成数据集信息文件](#43-生成数据集信息文件)
    - [5 离线推理](#5-离线推理)
        - [5.1 benchmark工具概述](#51-benchmark工具概述)
        - [5.2 离线推理](#52-离线推理)
    - [6 精度对比](#6-精度对比)
        - [6.1 310P离线推理精度](#61-310P离线推理精度)
        - [6.2 开源精度](#62-开源精度)
        - [6.3 精度对比](#63-精度对比)
    - [7 性能对比](#7-性能对比)
        - [7.1 性能对比](#71-性能对比)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[ReID论文](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1903.07071.pdf)  

### 1.2 代码地址
[ReID代码](https://github.com/michuanhaohao/reid-strong-baseline)  
branch:master  
commit_id: 3da7e6f03164a92e696cb6da059b1cd771b0346d

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN == 5.1.RC1
torch == 1.8.0
torchvision == 0.9.0
onnx == 1.7.0
```

### 2.2 python第三方库

```
decorator == 5.1.1
sympy == 1.10.1
numpy == 1.20.3
Pillow == 8.2.0
opencv-python == 4.5.2.54
yacs == 0.1.8
pytorch-ignite == 0.4.5
```


## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.获取，修改与安装ReID模型代码  

```
git clone https://github.com/michuanhaohao/reid-strong-baseline 
```

2.下载.pth权重文件

[pth权重文件](https://drive.google.com/open?id=1hn0sXLZ5yJcxtmuY-ItQfYD7hBtHwt7A)

[网盘pth权重文件，提取码：v5uh](https://pan.baidu.com/s/1ohWunZOrOGMq8T7on85-5w)  

文件名：market_resnet50_model_120_rank1_945.pth  
md5sum：0811054928b8aa70b6ea64e71ef99aaf


3.编写pth2onnx脚本ReID_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行ReID_pth2onnx.py脚本，生成onnx模型文件
```
python3.7 ReID_pth2onnx.py --config_file='reid-strong-baseline/configs/softmax_triplet_with_center.yml' MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('market_resnet50_model_120_rank1_945.pth')" TEST.NECK_FEAT "('before')" TEST.FEAT_NORM "('no')"  
```

 **模型转换要点：**  
> 加上TEST.NECK_FEAT "('before')" TEST.FEAT_NORM "('no')"导出的onnx可以推理测试性能  
> 不加上TEST.NECK_FEAT "('before')" TEST.FEAT_NORM "('no')"导出的onnx转换的om精度与官网精度一致

### 3.2 onnx转om模型

1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://gitee.com/link?target=https%3A%2F%2Fsupport.huawei.com%2Fenterprise%2Fzh%2Fdoc%2FEDOC1100164868%3FidPath%3D23710424%257C251366513%257C22892968%257C251168373)
需要指定输出节点以去除无用输出，可以使用netron开源可视化工具查看具体的输出节点名：

使用atc将onnx模型转为om, 【可通过npu-smi info指令查看npu使用情况】
![img_1.png](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

执行ATC命令
``` shell
atc --model=ReID.onnx \
--framework=5 \
--output=ReID_bs1 \
--input_format=NCHW \
--input_shape="image:1,3,256,128" \
--log=debug \
--soc_version=Ascend${chip_name} \
```
参数说明： <br>
--model：为ONNX模型文件。 <br>
--framework：5代表ONNX模型。 <br>
--output：输出的OM模型。 <br>
--input_format：输入数据的格式。<br> 
--input_shape：输入数据的shape。<br> 
--log：日志级别。 <br>
--soc_version：处理器型号。<br>

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
[Market1501数据集](http://www.liangzheng.org/Project/project_reid.html)

### 4.2 数据集预处理

执行两次预处理脚本ReID_preprocess.py，分别生成数据集query和数据集gallery预处理后的bin文件
```
export dataset_path=/opt/npu
python3.7 ReID_preprocess.py ${dataset_path}/market1501/query prep_dataset_query
python3.7 ReID_preprocess.py ${dataset_path}/market1501/bounding_box_test prep_dataset_gallery
mv prep_dataset_gallery/* prep_dataset_query/
```
### 4.3 生成数据集信息文件
执行gen_dataset_info.py脚本，生成数据集信息文件
```
python3.7 gen_dataset_info.py bin prep_dataset_query prep_bin.info 128 256
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息  
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://gitee.com/link?target=https%3A%2F%2Fsupport.huawei.com%2Fenterprise%2Fzh%2Fdoc%2FEDOC1100164868%3FidPath%3D23710424%257C251366513%257C22892968%257C251168373)
### 5.2 离线推理
1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./ReID_bs1.om -input_text_path=./prep_bin.info -input_width=128 -input_height=256 -output_binary=True -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_deviceX(X为对应的device_id)，每个输入对应一个_X.bin文件的输出，shape为bs*2048，数据类型为FP32.

## 6 精度对比

- **[310P离线推理精度](#61-310P离线推理精度)**  
- **[开源精度](#62-开源精度)**  
- **[精度对比](#63-精度对比)**  



### 6.1 310P离线推理精度
同310，调用ReID_postprocess.py脚本：
```
python3.7 ReID_postprocess.py --query_dir=${dataset_path}/market1501/query --gallery_dir=${dataset_path}/market1501/bounding_box_test --pred_dir=./result/dumpOutput_device0
```
310P精度结果：
```
RANK-1: 0.937
mAP: 0.858
```

### 6.2 开源精度
[原代码仓公布精度](https://github.com/michuanhaohao/reid-strong-baseline/blob/master/README.md)
```
Model	RANK-1   mAP
ReID	0.945     0.859 
```
### 6.3 精度对比
将得到的om离线模型推理IoU精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比
-   **[性能对比](#71-性能对比)**  


### 7.1 性能对比
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。  

使用benchmark工具在整个数据集上推理获得性能数据,可以获得吞吐率数据，推理后生成result/perf_vision_batchsize_{1}_device_0.txt：  
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./ReID_bs1.om -input_text_path=./prep_bin.info -input_width=128 -input_height=256 -output_binary=True -useDvpp=False
```



|           | 310      | 310P    | T4       | 310P/310 | 310P/T4  |
| --------- | -------- | ------- | -------- | -------- | -------- |
| bs1       | 1448.664 | 1292.33 | 1019.68  | 0.892084 | 1.267388 |
| bs4       | 1880.4   | 2505.71 | 1806.756 | 1.332541 | 1.386856 |
| bs8       | 2226.812 | 4035.45 | 2049.744 | 1.81221  | 1.968758 |
| bs16      | 2040.004 | 2397.23 | 2265.76  | 1.17511  | 1.058025 |
| bs32      | 2082.216 | 2274.7  | 2316.65  | 1.092442 | 0.981892 |
| bs64      | 2092.732 | 2292.27 | 2407.328 | 1.095348 | 0.952205 |
| 最优batch | 2226.812 | 4035.45 | 2407.328 | 1.81221  | 1.676319 |

最优的310P性能达到了最优的310性能的1.812倍，达到最优的T4性能的1.676倍。

#### 性能优化：性能已达标，不需要再优化
