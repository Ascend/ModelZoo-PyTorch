# RawNet2模型推理指导

- [RawNet2模型推理指导](#rawnet2模型推理指导)
	- [1 环境准备](#1-环境准备)
		- [1.1 下载pytorch源码](#11-下载pytorch源码)
		- [1.2 准备以下文件，放到pytorch源码根目录](#12-准备以下文件放到pytorch源码根目录)
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
git clone https://github.com/clovaai/aasist.git
```

### 1.2 准备以下文件，放到pytorch源码根目录
（1）该代码仓文件  
（2）[LA数据集](https://datashare.ed.ac.uk/handle/10283/3336) ，推理采用ASVspoof2019数据集的验证集进行精度评估。

### 1.3 安装依赖
安装 [onnx改图接口工具](https://gitee.com/peng-ao/om_gener)   

## 2 推理步骤
### 2.1 设置环境变量
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2.2 pt导出om模型
运行pth2om.sh导出om模型

${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/Ronnie_zheng/ascend-pytorch-crowdintelligence-doc/raw/master/Ascend-PyTorch%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%8C%87%E5%AF%BC/images/310P3.png)
```shell
bash pth2om.sh --model=aasist_bs1 --bs=1 --output_dir=output --soc_version=Ascend${chip_name} # Ascend310P3
```

### 2.3 om模型推理
```shell
python3 om_infer.py --om=aasist_bs1.om --batch=1
```

## 3 端到端推理Demo
提供run.sh，可直接执行
```shell
bash run.sh
```
