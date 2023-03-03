# Tdnn Onnx模型端到端推理指导
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
-   [6 精度对比](#6-精度对比)
	-   [6.1 离线推理精度统计](#61-离线推理精度统计)
	-   [6.2 精度对比](#62-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)
	-   [7.2 T4性能数据](#72-T4性能数据)
	-   [7.3 性能对比](#73-性能对比)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
华为npu模型训练组依据客户给的模型进行训练所得，无参考论文

### 1.2 代码地址
[Tdnn代码]https://gitee.com/ascend/modelzoo.git)  
branch:master  
commit_id=  
code_path=  



## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架 
```
pytorch = 昇腾1.5.0
torchaudio = 0.5.0
```
### 2.2 python第三方库 
```
speechbrain = 0.5.9
onnx == 1.7.0
onnxruntime-gpu == 1.8.1
sox == 14.4.0
```
1.gpu环境搭建  
在安装了Anaconda或Miniconda的gpu环境下，首先执行命令  
```
conda create -n tdnn python==3.7
```
创建名为tdnn的虚拟环境，再执行命令  
```
conda activate tdnn
```
进入该环境，先后执行下面两条命令    
```
pip3 install torch==1.5.0
pip3 install torchaudio==0.5.0
```
安装pytorch及torchaudio，再前往  
```
https://codeload.github.com/speechbrain/speechbrain/zip/refs/tags/v0.5.9
```
下载speechbrain 0.5.9源码包，解压并进入speechbrain项目根目录，在speechbrain/speechbrain文件夹下找到requirement.txt，删除torch和torchaudio对应行，然后执行  
```
pip install -r requirements.txt
pip install --editable .
```
完成speechbrain安装，接着执行  
```
pip install onnx==1.7.0
```
安装ONNX，执行  
```
pip install onnxruntime-gpu==1.7.0
```
安装onnxruntime-gpu

2.npu环境搭建  
在安装了Anaconda或Miniconda的npu环境下，首先执行命令  
```
conda create -n tdnn python==3.7
```
创建名为tdnn的虚拟环境，再执行命令  
```
conda activate tdnn
```
进入该环境，先执行    
```
pip3 install torch==1.5.0
```
安装pytorch，再尝试执行  
```
pip3 install torchaudio==0.5.0
```
安装torchaudio，确保安装成功后再前往  
```
https://codeload.github.com/speechbrain/speechbrain/zip/refs/tags/v0.5.9
```
下载speechbrain 0.5.9源码包，解压并进入speechbrain项目根目录，在speechbrain/speechbrain文件夹下找到requirement.txt，删除torch和torchaudio对应行，然后执行  
```
pip install -r requirements.txt
pip install --editable .
```
完成speechbrain安装  

**说明：** 
>  如果torchaudio安装失败，或者安装之后出现读取.flac文件报错的情况，    
>  请前往https://e.gitee.com/HUAWEI-ASCEND/notifications/infos?issue=I48AZM  
>  按步骤完成sox 14.4.0和torchaudio 0.5.0安装，再安装speechbrain    

## 3 模型转换

本模型基于开源框架PyTorch训练的TDNN模型进行转换。  
首先使用PyTorch将模型权重文件[tdnn.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/TDNN/PTH/classifier.ckpt

https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/TDNN/PTH/embedding_model.ckpt)转换为tdnn.onnx文件，再使用ATC工具将tdnn.onnx文件转为tdnn.om文件。  

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型  
以下步骤均在gpu环境完成。  
1.转换前的代码处理  
在上一步安装的speechbrain文件夹中，进入文件夹speechbrain/nnet，找到CNN.py文件，将第349行修改为  
```
padding_mode='constant'
```
再进入文件夹speechbrain/pretrained，用本仓库的interfaces.py替换该目录下的同名文件  

2.获取pth权重文件  
权重文件由华为npu模型训练组通过训练TDNN模型得到。  
在speechbrain/templates/speaker_id目录下新建文件夹best_model，将训练保存的模型文件及本仓库提供的hyperparams.yaml文件一并放到best_model。
best_model文件夹下应包含以下文件：
```
classifier.ckpt
hyperparams.yaml
embedding_model.ckpt
```

3.生成onnx模型文件  
将Tdnn_pth2onnx.py脚本放到speechbrain/templates/speaker_id目录下，并在该目录下执行
```
python Tdnn_pth2onnx.py
```
运行成功后，该目录下将生成tdnn.onnx文件  

 **说明：**  
>注意目前ATC支持的onnx算子版本为11


### 3.2 onnx转om模型

以下步骤在npu环境进行。  
1.生成om模型文件  
将atc.sh脚本放到speechbrain/templates/speaker_id目录下，并在该目录下执行。请以实际安装环境配置环境变量。
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash atc.sh tdnn.onnx tdnn
```
运行成功后，将生成tdnn.om模型  



## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
用户将自行获取的数据集上传到speechbrain/templates/speaker_id目录  

### 4.2 数据集预处理  
进入speechbrain/templates/speaker_id目录，将mini_librispeech_prepare.py文件的第174行代码random.shuffle(wav_list)注释掉，然后在该目录下执行  
```
python Tdnn_preprocess.py
```
预处理后的数据集在新生成的目录mini_librispeech_test_bin中  
### 4.3 生成数据集信息文件 
上一步的数据集预处理会同步生成一个文件mini_librispeech_test.info，该文件即数据集信息文件  



## 5 离线推理
1.设置离线推理环境  
将acl_net.py, pyacl_infer.py, om_infer.sh三个文件放到speechbrain/templates/speaker_id目录下  
2.执行pyacl离线推理  
在speechbrain/templates/speaker_id目录下执行  
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash om_infer.sh
```
推理结果将会输出到result目录下  



## 6 精度对比
-   **[离线推理精度](#61-离线推理精度)**  
-   **[精度对比](#62-精度对比)**  

### 6.1 离线推理精度统计

将Tdnn_postprocess.py文件放到speechbrain/templates/speaker_id目录下，并在该目录下执行  
```
python Tdnn_postprocess.py
```  
精度统计结果将直接输出到控制台    

### 6.2 精度对比
pth模型精度99.10%，om模型精度98.69%，模型转换后精度损失不超过1%   
 **精度调试：**  

>模型转换后精度损失不超过1%，精度达标，故不需要进行精度调试  



## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  
-   **[T4性能数据](#72-T4性能数据)**  
-   **[性能对比](#73-性能对比)**  

### 7.1 npu性能数据  
该模型不支持benchmark推理，故使用pyacl离线推理获取npu性能数据，npu性能数据为  
```
average pure infer time(ms):10.24
```
Interface throughputRate: 1000/10.24 = 97.37  
310单卡吞吐率：1000/10.54*16 = 1562.50 fps  

### 7.2 T4性能数据  
1.搭建环境  
在T4服务器搭建onnx模型推理环境，然后新建test文件夹，将下列文件上传至该文件夹  
```
tdnn.onnx文件  
Tdnn_onnx_infer.py文件  
speechbrain/templates/speaker_id目录下的mini_librispeech_test.info文件  
speechbrain/templates/speaker_id目录下的mini_librispeech_bin文件夹及其全部文件  
```
2.执行在线推理  
在test目录下执行  
```
python Tdnn_onnx_infer.py
```
性能数据将会输出到gpu_result目录下  
T4性能基线数据为  
```
average pure infer time(ms): 12.98  
```
T4单卡吞吐率：1000/12.98*16 = 1232.67 fps  

### 7.3 性能对比
单卡吞吐率  
```
npu-310：1562.50 fps  
gpu-t4 ：1232.67 fps  
```
310性能高于T4性能，性能达标  
  
 **性能优化：**  
>该模型性能优于T4，不用进行优化
