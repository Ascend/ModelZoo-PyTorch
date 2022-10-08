# RawNet2模型推理指导

-   [1 文件说明](#1-文件说明)
-   [2 环境准备](#2-环境准备)
	-   [2.1 文件下载](#21-文件下载)
	-   [2.2 文件拷贝](#22-文件拷贝)
	-   [2.3 设置环境变量](#23-设置环境变量)
-   [3 端到端推理步骤](#3-端到端推理步骤)
    -   [3.1 修改pytorch模型源码](#31-修改pytorch模型源码)
	-   [3.2 pth转onnx模型](#32-pth转onnx模型)
	-   [3.3 修改导出的onnx模型](#33-修改导出的onnx模型)
	-   [3.4 利用ATC工具转换为om模型](#34-利用ATC工具转换为om模型)
	-   [3.5 om模型推理](#35-om模型推理)

------

## 1 文件说明
```
RawNet2_for_Pytorch  
  ├── pth2onnx.py       pytorch模型导出onnx模型  
  ├── modify_onnx.py    修改导出的onnx模型  
  ├── atc.sh            onnx模型转om  
  ├── om_infer.py       推理导出的om模型  
  └── acl_net.py        PyACL推理工具代码  
```

## 2 环境准备

### 2.1 文件下载
- [RawNet2_Pytorch源码下载](https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2)
  ```
  git clone https://github.com/asvspoof-challenge/2021.git
  cd 2021/LA/Baseline-RawNet2/
  ```
- [权重下载](https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip)
- [数据集下载](https://datashare.ed.ac.uk/handle/10283/3336)  
  om推理采用ASVspoof2019数据集的验证集进行精度评估。

### 2.2 文件拷贝
拷贝pth2onnx.py，modify_onnx.py，atc.sh，om_infer.py，acl_net.py文件到2021/LA/Baseline-RawNet2/目录下。  
将下载的权重文件pre_trained_DF_RawNet2.pth放在和代码同一目录下。  
在同一目录下创建data目录并将下载的数据集放入，data目录中的文件结构如下所示。
```
data  
  └── LA  
    ├── ASVspoof2019_LA_asv_protocols
    ├── ASVspoof2019_LA_asv_scores
    ├── ASVspoof2019_LA_cm_protocols
    ├── ASVspoof2019_LA_dev
    ├── ASVspoof2019_LA_eval
    └── ASVspoof2019_LA_train
```

### 2.3 设置环境变量
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 3 端到端推理步骤

### 3.1 修改pytorch模型源码
导onnx模型需要修改2021/LA/Baseline-RawNet2/model.py中的SincConv类，在该类的forward函数中增加一行，如下所示。
```python
self.band_pass = torch.from_numpy(self.band_pass.numpy())   # 增加行，和下行缩进保持一致
band_pass_filter=self.band_pass.to(self.device)  # 根据该行代码找到增加位置
```

### 3.2 pth导出onnx 
```python
python3.7 pth2onnx.py \
        --pth_model=pre_trained_DF_RawNet2.pth \
        --onnx_model=rawnet2_ori.onnx \
        --batch_size=1
```

### 3.3 修改导出的onnx模型
```python
python3.7 -m onnxsim rawnet2_ori.onnx rawnet2_sim.onnx

python3.7 modify_onnx.py \
        --input_onnx=rawnet2_sim.onnx \
        --output_onnx=rawnet2_modify.onnx
```

### 3.4 利用ATC工具转换为om模型
```shell
bash atc.sh rawnet2_modify.onnx rawnet2_modify input:1,64600
```
注：目前ATC支持的onnx算子版本为11

### 3.5 om模型推理
```python
python3.7 om_infer.py \
    --batch_size=1 \
    --om_path=rawnet2_modify.om \
    --eval_output='rawnet2_modify_om.txt' \
    --database_path='data/LA/' \
    --protocols_path='data/LA/'
```