# SPACH_SMLP模型PyTorch离线推理指导

- [SPACH_SMLP模型PyTorch离线推理指导](#spach_smlp模型pytorch离线推理指导)
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
  - [5 离线推理](#5-离线推理)
    - [5.1 获取ais_infer推理工具](#51-获取ais_infer推理工具)
    - [5.2 离线推理](#52-离线推理)
  - [6 精度对比](#6-精度对比)
    - [6.1 离线推理精度统计](#61-离线推理精度统计)
    - [6.2 开源精度](#62-开源精度)
    - [6.3 精度对比](#63-精度对比)
  - [7 性能对比](#7-性能对比)
    - [7.1 npu性能数据](#71-npu性能数据)
    - [7.2 gpu，npu推理性能对比](#72-gpunpu推理性能对比)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[SPACH_SMLP论文](https://arxiv.org/abs/2109.05422)  

### 1.2 代码地址
[SPACH_SMLP代码](https://github.com/microsoft/SPACH)  
branch:master  
commit_id:b11b098970978677b7d33cc3424970152462032d

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.1.RC1
torch==1.5.0
torchvision==0.2.2
onnx==1.8.0
onnxruntime==1.9.0
```

### 2.2 python第三方库

```
timm==0.3.2
einops==0.3.2
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.准备pth权重文件  
使用训练好的pth权重文件

2.使用开源仓，获取开源命令

```
git clone git@github.com:microsoft/SPACH.git
cd SPACH  
git reset b11b098970978677b7d33cc3424970152462032d --hard
```
3.SPACH代码迁移，执行命令。
```
git clone https://gitee.com/ascend/ModelZoo-PyTorch
cd ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/classfication/SPACH-SMLP  
```

4.调用“SPACH-SMLP”目录中的“smlp_pth2onnx.py”脚本导出ONNX模型。

sMLP预训练[PyTorch模型权重文件](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/sMLP/smlp_t.pth),[ONNX模型文件](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/sMLP/sMLPNet-T.onnx)

```
python smlp_pth2onnx.py --onnx_path sMLPNet-T.onnx --model_name smlpnet_tiny  --pth_path smlp_t.pth  --opset_version 11
```
参数说明：

--model_name MODEL    模型名称

--pth_path PTH_PATH    pytorh模型路径

--onnx_path ONNX_PATH    ONNX模型路径

--opset_version OPSET_VERSION  ONNX opset版本，默认11

 **说明：**  
>注意目前ATC支持的onnx算子版本为11


### 3.2 onnx转om模型

1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN 开发辅助工具指南 (推理) 01](https://www.hiascend.com/document/detail/zh/canncommercial/51RC1/inferapplicationdev/atctool) 
${chip_name}可通过npu-smi info指令查看，例：310P3

![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```
atc --framework=5 --model=sMLPNet-T.onnx --output=sMLPNet-T-batch8-high --input_format=NCHW --input_shape="input:8,3,224,224" --soc_version=Ascend${chip_name} --op_precision_mode=op_precision.ini
```


参数说明：

--model：为ONNX模型文件。  

--framework：5代表ONNX模型。  

--input_format：输入数据的格式。  

--input_shape：输入数据的shape。  

--output：输出的OM模型。  

--log：日志级别。  

--soc_version：处理器型号。 

--op_precision_mode：转换模式，开启GELU高性能模式。

运行成功后生成“sMLPNet-T-batch8-high.om”模型文件。

> 可从OBS处直接下载batch size为[1](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/sMLP/sMLPNet-T-batch1-high.om)、[4](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/sMLP/sMLPNet-T-batch4-high.om)、[8](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/sMLP/sMLPNet-T-batch8-high.om)、[16](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/sMLP/sMLPNet-T-batch16-high.om)、[32](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/sMLP/sMLPNet-T-batch32-high.om)、[64](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/sMLP/sMLPNet-T-batch64-high.om)的om模型
## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
 用户自行获取原始数据集，可选用的开源数据集包括ImageNet201，将数据集上传到服务器任意路径下并解压。

   以ImageNet2012数据集为例，数据集目录结构参考如下所示。

   ```
   ├── ImageNet2012
         ├──train
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...   
              ├──...                     
         ├──val  
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...              
   ```

### 4.2 数据集预处理
将原始数据（.jpg）转化为二进制文件（.bin）。转化方法参考mmdetection预处理方法，以获得最佳精度。以coco_2017数据集为例，通过缩放、均值方差手段归一化，输出为二进制文件。

执行“smlp_preprocess.py”脚本。

```
python3.7 smlp_preprocess.py --save_dir imagenet-val-bin --data_root /opt/npu/imagenet/
```

参数说明：

--batch_size BATCH_SIZE     批处理大小，

--data_root DATA_ROOT    数据集路径

--save_dir SAVE_DIR   处理后的数据集路径

每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成“imagenet-val-bin”二进制文件夹。

## 5 离线推理

-   **[获取ais_infer推理工具](#51-获取ais_infer推理工具)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 获取ais_infer推理工具

https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer

将工具编译后的压缩包放置在当前目录；解压工具包，安装工具压缩包中的whl文件； pip3 install aclruntime-0.01-cp37-cp37m-linux_xxx.whl

### 5.2 离线推理
昇腾芯片上执行，执行时使npu-smi info查看设备状态，确保device空闲

1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理

执行推理
```
python3.7.5 ais_infer.py  --model /home/infname63/spach-smlp/sMLPNet-T-batch1-high.om  --batchsize 1 --output ./ --outfmt BIN --loop 100 
```
参数说明:

--model：模型地址

--input：预处理完的数据集文件夹

--output：推理结果保存地址


## 6 精度对比

-   **[离线推理mAP精度](#61-离线推理mAP精度)**  
-   **[开源mAP精度](#62-开源mAP精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度统计

后处理统计精度

调用smlp_postprocess.py脚本与label比对，可以获得Accuracy Top1，Top5 准确率数据。

```
python smlp_postproces.py --infer_result_dir ${infer_result_dir}
```

参数说明:
	
--infer_result_dir参数：为ais_infer.py运行后生成的存放推理结果的目录的路径。 例如本例中为~/spach-smlp/ais_infer/2022_07_09-18_05_40/

NPU精度如下：

```
Model         top1 acc     
SPACH_SMLP        81.25
```
### 6.2 开源精度
[开源代码仓精度](https://github.com/microsoft/SPACH)

```
Model         top1 acc     
SPACH_SMLP        81.9  
```
### 6.3 精度对比
将得到的om离线模型推理精度与该模型github代码仓上公布的Top1 acc对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
1.ais_infer工具在整个数据集上推理获得性能数据  
batch1的性能，ais_infer工具在整个数据集上推理后生成result/sumary.json：  
```
"NPU_compute_time": {
   "min": 55.18075942993164, 
   "max": 55.18075942993164, 
   "mean": 55.18075942993164, 
   "median": 55.18075942993164, 
   "percentile(99%)": 55.18075942993164}
```
Interface throughputRate:1000 * batchsize/npu_compute_time.mean= 290.0 既是batch1 310P单卡吞吐率

### 7.2 gpu，npu推理性能对比

| batchsize | ascend-310p | GPU-t4 |
| --------- | ----------- | ------ |
| 1         | 171.6       | 177.7  |
| 4         | 273.5       | 341.5  |
| 8         | 298.7       | 359.0  |
| 16        | 290.0       | 363.7  |
| 32        | 273.0       | 371.0  |
| 64        | 257.5       | 359.1  |
| best      | 298.7       | 371.0  |

sMLP模型在NPU上的性能是GPU上最优性能的0.805倍

> 注：已通过性能评审

