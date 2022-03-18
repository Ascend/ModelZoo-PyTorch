# LSTM Onnx模型端到端推理指导
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
npu训练组依据客户给的模型进行训练所得，无参考论文

### 1.2 代码地址
[LSTM代码]https://gitee.com/ascend/modelzoo.git)  
branch:master  
commit_id=8ed54e7d0fc9b632e1e3b9420bed96ee2c7fa1e3  
code_path=modelzoo/tree/master/built-in/PyTorch/Official/nlp/LSTM_for_PyTorch

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
pytorch = 1.8.0
torchvision = 0.9.0
```
### 2.2 python第三方库

```
ONNX == 1.7.0
kaldi == https://github.com/kaldi-asr/kaldi
Pillow == 7.2.0
onnxruntime-gpu == 1.7.0
kaldiio == 2.17.2
```
kaldi需要安装在ModelZoo的LSTM源码仓中“modelzoo/built-in/PyTorch/Official/nlp/LSTM_for_PyTorch/NPU/1p/”目录下。ModelZoo的LSTM源码仓下载方法.
```
git clone https://gitee.com/ascend/modelzoo.git
cd modelzoo
git reset --hard 8ed54e7d0fc9b632e1e3b9420bed96ee2c7fa1e3
```
1.下载ModelZoo的LSTM源码仓
```
git clone https://gitee.com/ascend/modelzoo.git
cd modelzoo
git reset --hard 8ed54e7d0fc9b632e1e3b9420bed96ee2c7fa1e3
cd built-in/PyTorch/Official/nlp/LSTM_for_PyTorch/NPU/1p/
```
2.下载kaldi工具包
源码搭建kaldi工具包环境。以arm 64位环境为例说明，推荐安装至conda环境：
```
git clone https://github.com/kaldi-asr/kaldi
cd kaldi
```
3.检查工具包所需依赖并安装缺少依赖
```
tools/extras/check_dependencies.sh
```
根据检查结果和提示，安装缺少的依赖。安装完依赖再次检查工具包所需依赖是否都安装ok
4.编译
```
cd tools
make -j 64
```
5.安装依赖库成功之后安装第三方工具，Kaldi使用FST作为状态图的表现形式，安装方式如下：
```
make openfst
extras/install_irstlm.sh
extras/install_openblas.sh
```

```
输出：Installation of IRSTLM finished successfully
输出：OpenBLAS is installed successfully
```
6.配置源码
```
cd ../src/
./configure --shared
输出"Kaldi has been successfully configured."
```
7.编译安装
```
make -j clean depend
make -j 64

输出：echo Done
Done
```
源码中使用的python2.7版本，如果系统python版本与该版本不同，可使用系统默认python，在目录kaldi/python/下创建空文件.use_default_python。其他安装问题可参见kaldi官方安装教程.

**说明：** 
>  将源码包中的全部脚本移动到已安装kaldi工具的“modelzoo/built-in/PyTorch/Official/nlp/LSTM_for_PyTorch/NPU/1p/”目录下。
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型
1.下载pth权重文件  
权重文件由华为npu模型训练组提供。
2.lstm模型代码在代码仓中
```
git clone https://gitee.com/ascend/modelzoo.git
```
 3.编写pth2onnx脚本LSTM_pth2onnx.py
本模型基于开源框架PyTorch训练的lstm进行模型转换。使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。权重文件由npu模型训练组提供，gpu训练模型ctc_best_model.pth。源码包中已提供ctc_best_model.pth权重文件。在1p目录下创建checkpoint/ctc_fbank_cnn/目录并将权重文件移到到该目录下。
```
mkdir -p checkpoint/ctc_fbank_cnnmv ./ctc_best_model.pth ./checkpoint/ctc_fbank_cnn/
```

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 ./steps/LSTM_pth2onnx.py --batchsize=16
```

### 3.2 onnx转om模型

1.修改lstm_atc.sh脚本，通过ATC工具使用脚本完成转换，具体的脚本示例如下：
```
# 配置环境变量
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01
```
atc --input_format=ND --framework=5 --model=lstm_ctc_16batch.onnx --input_shape="actual_input_1:16,390,243" --output=lstm_ctc_16batch_auto --auto_tune_mode="RL,GA" --log=info --soc_version=Ascend310
```
参数说明：
   --model：为ONNX模型文件。
   --framework：5代表ONNX模型。
   --output：输出的OM模型。
   --input_format：输入数据的格式。
   --input_shape：输入数据的shape。
   --log：日志级别。
   --soc_version：处理器型号。
   
执行lstm_atc.sh脚本，将.onnx文件转为离线推理模型文件.om文件。
```
bash lstm_atc.sh
```
运行成功后生成lstm_ctc_npu_16batch.om用于二进制输入推理的模型文件。

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
本模型支持timit语音包的验证集。timit数据集与训练对齐，使用训练提供的语音数据包。需用户自行获取数据集，并将数据集命名为data.zip，并上传数据集data.zip至服务器ModelZoo的LSTM源码仓下的built-in/PyTorch/Official/nlp/LSTM_for_PyTorch/NPU/目录中。数据集结构如下。
```
├── DOC
├── README.DOC
├── TEST
└── TRAIN
```

### 4.2 数据集预处理
目的、处理过程及方法、处理后输出文件介绍及用途模型输入数据为二进制格式。将原始数据（audio数据）转化为二进制文件（.bin）。
1.解压数据集
```
unzip data.zip
cd p1
```
2.修改1p目录下path.sh里第一行代码如下：
```
KALDI_ROOT=./kaldi
```
3.创建data文件夹
```
mkdir data
```
4.执行prepare_data.sh脚本。
```
chmod +x local/timit_data_prep.sh
chmod +x steps/make_feat.sh
bash prepare_data.sh
```
执行prepare_data.sh脚本之后，在当前目录下会生成tmp文件夹和在data文件夹下生成dev,test,train三个数据集文件夹。
5.移动LSTM_preprocess_data.py至1p/steps目录下，
6.修改./conf/ctc_config.yaml文件内容
```
#[test]
test_scp_path: 'data/dev/fbank.scp'
test_lab_path: 'data/dev/phn_text'
decode_type: "Greedy"
beam_width: 10
lm_alpha: 0.1
lm_path: 'data/lm_phone_bg.arpa'
```
data文件夹即为执行prepare_data.sh之后所生成，使用此目录下的dev数据集进行验证。
7.执行LSTM_preprocess_data.py脚本
```
python3.7 ./steps/LSTM_preprocess_data.py --conf=./conf/ctc_config.yaml --batchsize=16
```
参数为配置文件。
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本LSTM_get_info.py
使用pyacl推理需要输入二进制数据集的info文件，用于获取数据集。使用LSTM_get_info.py脚本，输入已经得到的二进制文件，输出生成二进制数据集的info文件。上传LSTM_get_info.py至1p文件夹下，运行LSTM_get_info.py脚本。

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 LSTM_get_info.py --batchsize=16
```
参数为om模型推理batchsize。运行成功后，在当前目录中生成lstm.info。
## 5 离线推理
1.配置pyacl推理环境变量
```
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/pyACL/python/site-packages/acl:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```
2.执行pyacl离线推理
```
#pyacl推理命令
python3.7 ./pyacl_infer.py --model_path=./lstm_ctc_16batch_auto.om --device_id=0 --cpu_run=True --sync_infer=True --workspace=10 --input_info_file_path=./lstm.info --input_dtypes=float32 --infer_res_save_path=./infer_res --res_save_type=npy
```
参数说明：
--model_path：om模型文件
--input_info_file_path：处理后的数据集信息info文件
--infer_res_save_path：pyacl推理后结果保存路径
--res_save_type：推理结果保存格式，npy格式为含有shape信息的数据，bin格式为不含shape-信息的二进制numpy数据


## 6 精度对比
-   **[离线推理精度](#61-离线推理精度)**  
-   **[精度对比](#62-精度对比)**  

### 6.1 离线推理精度统计

1. 后处理统计精度
上传LSTM_postprocess_data.py脚本至1p/steps目录下，执行LSTM_postprocess_data.py脚本进行数据后处理。
```
python3.7 ./steps/LSTM_postprocess_data.py --conf=./conf/ctc_config.yaml --npu_path=./infer_res/ --batchsize=16
```
conf参数为模型配置文件, npu_path参数为pyacl推理结果目录。执行后处理脚本之后，精度数据由WER 与CER给出，字母错误率与单词错误率。
```
Character error rate on test set: 13.5877
Word error rate on test set: 18.9075
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上   

### 6.2 精度对比
推理模型om精度与onnx精度一致，且与训练测试pth模型精度一致。  
 **精度调试：**  

>没有遇到精度不达标的问题，故不需要进行精度调试  

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  
-   **[T4性能数据](#72-T4性能数据)**  
-   **[性能对比](#73-性能对比)**  

### 7.1 npu性能数据
1.该模型不支持benchmark推理，使用pyacl推理获取npu性能数据。
batch1的性能
```
Bs1: average pure infer time(ms):179.183
```
Interface throughputRate: 1000/179.183 = 5.58，5.58x4既是batch1 310单卡吞吐率  
batch4的性能  
```
Bs4: average pure infer time(ms):187.361
```
Interface throughputRate: 1000/187.361* 4 = 21.35，21.35*4既是batch4 310单卡吞吐率  
batch8性能
```
Bs8: average pure infer time(ms):202.751
```
batch8 310单卡吞吐率：1000/202.751 * 8 = 157.83 fps  
batch16性能：
```
Bs16: average pure infer time(ms):195.763
```
batch16 310单卡吞吐率：1000/195.763 * 16 * 4 = 326.93fps  
batch32性能：
```
Bs32: average pure infer time(ms):260.119
```
batch32 310单卡吞吐率：1000/260.119 * 32 * 4 = 492.08fps  

### 7.2 T4性能数据
gpu下onnx在线推理获取T4性能基线
在T4环境下搭建环境，将预处理好的bin文件数据打包和lstm_infer_onnx.py脚本上传至服务器上1p目录下，进行onnx在线推理，执行lstm_onnx_infer.py脚本.
```
python3.7 lstm_onnx_infer.py --conf=./conf/ctc_config.yaml --model_path=./lstm_ctc_16batch.onnx --bin_file_path=./lstm_bin/ --pred_res_save_path=./lstm_onnx_infer --batchsize=16
```
性能基线数据为：
```
total infer time(ms): 2308.3143849999997
average infer time(ms): 92.3325754
```
batch16 t4吞吐率：1000/92.33 * 16 = 173.29fps 

### 7.3 性能对比
batch16： 326.93 >1000/(92.33/16)  

310单个device的吞吐率乘4即单卡吞吐率比T4单卡的吞吐率大，故310性能高于T4性能，性能达标。  
  
 **性能优化：**  
>该模型性能优于T4，不用进行优化
