# Shell 脚本 说明

**必要目录结构及说明 （Ascend310）**

```
|—— LResNet						# 源码目录
|-- LResNet.patch				# 源码补丁
|-- test						
|------ eval_acc_perf.sh		# 预处理推理后处理一条龙
|------ pth2om.sh				# pth转onnx，onnx转om脚本
|—— data
|------ lfw.bin					# lfw数据集
|—— model
|——---- model_ir_se100.pth		# 模型权重
|—— LResNet_postprocess.py		# 后处理脚本
|—— LResNet_preprocess.py		# 预处理脚本
|—— LResNet_pth2onnx.py			# pth转onnx脚本
|-- benchmark.x86_64			# benckmark工具
```

**step1：准备阶段修改源码**

```bash
cd LResNet
patch -p1 < ../LResNet.patch
rm -rf ./work_space/* 
mkdir ./work_space/history && mkdir ./work_space/log && mkdir ./work_space/models && mkdir ./work_space/save
cd ..
```

**step2：获取模型权重，并放在工作目录的model文件夹下**

OBS： [model_ir_se100.pth](obs://l-resnet100e-ir/eval/model_ir_se100.pth)  云盘：[model_ir_se100.pth](https://drive.google.com/file/d/1rbStth01wP20qFpot06Cy6tiIXEEL8ju/view?usp=sharing)

**step3：获取LFW数据集，放在工作目录的data目录下**

OBS： [lfw.bin](obs://l-resnet100e-ir/eval/lfw.bin) 云盘： [lfw.bin](https://drive.google.com/file/d/1mRB0A8f0b5GhH7w0vNMGdPjSWF-VJJLY/view?usp=sharing) 



**1.pth转om模型**

```shell
bash test/pth2om.sh
```

**2.npu性能数据及精度数据**

```shell
bash test/eval_acc_perf.sh --data_path='./data/lfw.bin'
```

**必要目录结构及说明 （t4）**

onnx模型权重由第一步 pth转om 模型生成在 model 文件夹下

```
|-- test						
|------ pref_gpu.sh							# onnx性能数据脚本
|—— model
|——---- model_ir_se100_bs1_sim.onnx			# bs=1 模型权重
|——---- model_ir_se100_bs16_sim.onnx		# bs=16 模型权重
|-- trtexec									# trtexec工具
```

**3.测试t4性能数据**

```
bash test/pref_gpu.sh
```

