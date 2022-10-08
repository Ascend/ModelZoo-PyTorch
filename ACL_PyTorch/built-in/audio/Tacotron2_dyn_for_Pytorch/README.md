# Tacotron2_dyn模型推理指导

- [Tacotron2_dyn模型推理指导](#Tacotron2_dyn模型推理指导)
	- [1 环境准备](#1-环境准备)
		- [1.1 下载pytorch源码](#11-下载pytorch源码)
		- [1.2 准备以下文件，放到pytorch Tacotron2源码目录](#12-准备以下文件放到pytorch Tacotron2源码目录)
		- [1.3 安装依赖](#13-安装依赖)
	- [2 推理步骤](#2-推理步骤)
		- [2.1 设置环境变量](#21-设置环境变量)
		- [2.2 pt导出om模型](#22-pt导出om模型)
		- [2.3 om模型推理](#23-om模型推理)
	- [3 端到端推理Demo](#3-端到端推理demo)

------


## 1 环境准备

### 1.1 下载pytorch源码
```shell
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples
git reset --hard 7ce175430ff9af25b040ffe2bceb5dfc9d2e39ad
cd PyTorch/SpeechSynthesis/Tacotron2
mkdir -p output/audio
mkdir checkpoints
```

### 1.2 准备以下文件，放到pytorch Tacotron2源码目录
（1）代码文件
```shell
Tacotron2_dyn_for_PyTorch
├── cvt_tacotron2onnx.py  放到Tacotron2/tensorrt下
├── cvt_waveglow2onnx.py  放到Tacotron2/tensorrt下
├── atc.sh        放到Tacotron2下
├── acl_infer.py  放到Tacotron2下
├── val.py        放到Tacotron2下
└── run.sh        放到Tacotron2下
```
（2）模型文件  
`nvidia_tacotron2pyt_fp32_20190427`：[下载地址](https://ngc.nvidia.com/catalog/models/nvidia:tacotron2_pyt_ckpt_fp32)  
`nvidia_waveglowpyt_fp32_20190427`：[下载地址](https://ngc.nvidia.com/catalog/models/nvidia:waveglow_ckpt_fp32)  
将下载的模型文件`nvidia_tacotron2pyt_fp32_20190427`和`nvidia_waveglowpyt_fp32_20190427`放在新建的`checkpoints`文件夹下

### 1.3 安装依赖
安装依赖包：`pip install -r requirements.txt`  
安装 [om推理接口工具](https://gitee.com/peng-ao/pyacl) 

## 2 推理步骤
### 2.1 设置环境变量
根据实际安装的`CANN`包路径修改`/usr/local/Ascend/ascend-toolkit`。
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2.2 pt导出om模型
`pytorch`导出`onnx`，进一步导出`om`文件，结果默认保存在`output`文件夹下。
```shell
# pytorch导出onnx
python3 tensorrt/cvt_tacotron2onnx.py --tacotron2 ./checkpoints/nvidia_tacotron2pyt_fp32_20190427 -o output/ -bs ${bs}
python3 tensorrt/cvt_waveglow2onnx.py --waveglow ./checkpoints/nvidia_waveglowpyt_fp32_20190427 -o output/ --config-file config.json

# onnx导出om
bash atc.sh ${SOC_VERSION} ${bs} 256
```
其中，`bs`为模型`batch_size`，`SOC_VERSION`可通过`npu-smi info`命令查看。
![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

### 2.3 om模型推理
运行`val.py`推理`om`模型，合成语音默认保存在`output/audio`文件夹下。
```shell
# 推理tacotron2 om
python3 val.py -i filelists/ljs_audio_text_test_filelist.txt -bs ${bs} -device_id ${device_id}

# 推理waveglow生成wav文件
python3 val.py -i filelists/ljs_audio_text_test_filelist.txt -o output/audio -bs ${bs} -device_id ${device_id} --gen_wav
```
其中，`bs`为模型`batch_size`，`device_id`设置推理用第几号卡。

## 3 端到端推理Demo
提供`run.sh`，作为实现端到端推理的样例参考。
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash run.sh Ascend310P3 4 0  # ${SOC_VERSION} ${bs} ${device_id}
```
