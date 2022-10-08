# Shell 脚本 说明

**必要目录结构及说明 （Ascend310）**

```
|—— TokenLabeling				# 源码目录
|-- LV-Vit.patch				# 源码补丁
|-- test						
|------ eval_acc_perf.sh		# 预处理推理后处理一条龙
|------ pth2om.sh				# pth转onnx，onnx转om脚本
|—— data		                # 用于存放imagenet验证集二进制文件
|------ val.txt                 # imagenet 纯验证集标签
|—— model
|——---- model_best.pth.tar		# 模型权重
|—— LV_Vit_postprocess.py		# 后处理脚本
|—— LV_Vit_preprocess.py		# 预处理脚本
|—— LV_Vit_pth2onnx.py			# pth转onnx脚本
|-- benchmark.x86_64			# benckmark工具
```

**step1：准备阶段修改源码**

```bash
git clone https://github.com/zihangJiang/TokenLabeling.git
cd TokenLabeling
patch -p1 < ../LV-Vit.patch
cd ..
```

**step2：获取模型权重，并放在工作目录的model文件夹下**

```bash
wget https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_s-26M-224-83.3.pth.tar

mv lvvit_s-26M-224-83.3.pth.tar ./model/model_best.pth.tar
```

**step3：获取imagenet纯验证数据集，放在该目录**

/opt/npu/imagenet/PureVal/


**1.pth转om模型**

```shell
bash test/pth2om.sh
```

**2.npu性能数据及精度数据**

--datasets_path=imagenet纯验证集路径

```shell
bash test/eval_acc_perf.sh --datasets_path=/opt/npu/imagenet/PureVal/
```

**必要目录结构及说明 （t4）**

onnx模型权重由第一步 pth转om 模型生成在 model 文件夹下
请自行软链接trtexec工具！

```
|-- test						
|------ pref_gpu.sh							# onnx性能数据脚本
|—— model
|——---- model_best_bs1_sim.onnx			    # bs=1 模型权重
|——---- model_best_bs16_sim.onnx		    # bs=16 模型权重
|-- trtexec									# trtexec工具
```

**3.测试t4性能数据**

```
bash test/pref_gpu.sh
```

