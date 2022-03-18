# LV-Vit Onnx模型端到端推理指导

+ [1模型概述](#1 模型概述)

  + [1.1 论文地址](##1.1 论文地址)
  + [1.2 代码地址](##1.2 代码地址)

+ [2 环境说明](#2 环境说明)

  + [2.1 深度学习框架](##2.1 深度学习框架)
  + [2.2 python第三方库](##2.2 python第三方库)

+ [3 模型转换](#3 模型转换)

  + [3.1 pth转onnx模型](##3.1 pth转onnx模型)
  + [3.2 onnx转om模型](##3.2 onnx转om模型)

+ [4 数据集预处理](#4 数据集预处理)

  + [4.1 数据集获取](##4.1 数据集获取)
  + [4.2 数据集预处理](##4.2 数据集预处理)
  + [4.3 生成预处理数据集信息文件](##4.3 生成预处理数据集信息文件)

+ [5 离线推理](#5 离线推理)

  + [5.1 benchmark工具概述](##5.1 benchmark工具概述)
  + [5.2 离线推理](##5.2 离线推理)

+ [6 精度对比](#6 精度对比)

  + [6.1 离线推理精度统计](##6.1 离线推理精度统计)
  + [6.2 开源精度](##6.2 开源精度)
  + [6.3 精度对比](##6.3 精度对比)

+ [7 性能对比](#7 性能对比)

  + [7.1 npu性能数据](##7.1 npu性能数据)
  + [7.2 gpu和npu性能对比](##7.2 gpu和npu性能对比)

  

## 1 模型概述

### 1.1 论文地址

[LV-Vit论文](https://arxiv.org/abs/2104.10858 )

### 1.2 代码地址

[LV-Vit代码](https://github.com/zihangJiang/TokenLabeling )



## 2 环境说明

### 2.1 深度学习框架

```
torch==1.8.0
torchvision==0.9.0
onnx==1.10.1
onnx-simplifier==0.3.6
```

### 2.2 python第三方库

```
numpy==1.21.2
pyyaml==5.4.1
pillow==8.3.1
timm==0.4.5
scipy==0.24.2
```



## 3 模型转换

### 3.1 pth转onnx模型

1.LV-Vit模型代码下载

```bash
# 切换到工作目录
cd LV-Vit

git clone https://github.com/zihangJiang/TokenLabeling.git
cd TokenLabeling
patch -p1 < ../LV-Vit.patch
cd ..
```

2.获取模型权重，并放在工作目录的model文件夹下
在model/下已经存放了在gpu8p上训练得到的pth，如需下载官方pth，则执行以下代码
```bash
wget https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_s-26M-224-83.3.pth.tar
mv lvvit_s-26M-224-83.3.pth.tar model_best.pth.tar

rm ./model/model_best.pth.tar
mv model_best.pth.tar ./model/
```



3.使用 LV_Vit_pth2onnx.py 脚本将pth模型文件转为onnx模型文件

+ 参数1：pth模型权重的路径

+ 参数2：onnx模型权重的存储路径

+ 参数3：batch size

```bash.
python LV_Vit_pth2onnx.py ./model/model_best.pth.tar ./model/model_best_bs1.onnx 1
python LV_Vit_pth2onnx.py ./model/model_best.pth.tar ./model/model_best_bs16.onnx 16
```

4.使用 onnxsim 工具优化onnx模型

+ 参数1：输入的shape
+ 参数2：onnx模型权重的存储路径
+ 参数3：优化后onnx模型权重的存储路径

```
python -m onnxsim --input-shape="1,3,224,224" ./model/model_best_bs1.onnx ./model/model_best_bs1_sim.onnx
python -m onnxsim --input-shape="16,3,224,224" ./model/model_best_bs16.onnx ./model/model_best_bs16_sim.onnx
```

5.使用tensorRT工具测试onnx模型性能

请自行软链接trtexec工具

```
./trtexec --onnx=model/model_best_bs1_sim.onnx --fp16 --shapes=image:1x3x112x112 --device=0 > sim_onnx_bs1.log
./trtexec --onnx=model/model_best_bs16_sim.onnx --fp16 --shapes=image:16x3x112x112 --device=0 > sim_onnx_bs16.log
```



### 3.2 onnx转om模型

1.设置环境变量

```bash
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

2.使用 atc 将 onnx 模型转换为 om 模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)

请注意，为了优化softmax算子，在其前后添加了transpose算子，故一并优化transpose，须在白名单中添加（batch_size，6，197，197）和
（batch_size，197，197，6）

路径：/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/opp/op_impl/built-in/ai_core/tbe/impl/dynamic/transpose.py

```bash
atc --framework=5 --model=./model/model_best_bs1_sim.onnx --output=./model/model_best_bs1_sim --input_format=NCHW --input_shape="image:1,3,224,224" --log=debug --soc_version=Ascend310

atc --framework=5 --model=./model/model_best_bs16_sim.onnx --output=./model/model_best_bs16_sim --input_format=NCHW --input_shape="image:16,3,224,224" --log=debug --soc_version=Ascend310
```



## 4 数据集预处理

### 4.1 数据集获取

获取imagenet纯验证数据集，放在该目录：/opt/npu/imagenet/PureVal/



### 4.2 数据集预处理

执行预处理脚本，会在工作目录的data目录下生成数据集预处理后的 bin 文件和 数据集信息文件

LV_Vit_preprocess.py：
+ --src_path: imagenet纯验证集路径; --save_path: bin文件存放路径

gen_dataset_info.py
+ 参数1：bin文件
+ 参数2：数据bin文件存放目录

```
python LV_Vit_preprocess.py --src_path /opt/npu/imagenet/PureVal/ --save_path ./data/prep_dataset;
python gen_dataset_info.py ./data/prep_dataset ./data/lvvit_prep_bin.info;
```


## 5 离线推理

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)

### 5.2 离线推理

1.设置环境变量

```bash
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest/
```

2.执行离线推理, 输出结果默认保存在当前目录result/dumpOutput_device0

```bash
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./model/model_best_bs1_sim.om -input_text_path=lvvit_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False

./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=./model/model_best_bs16_sim.om -input_text_path=lvvit_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```



## 6 精度对比

### 6.1 离线推理精度统计

执行后处理脚本统计om模型推理结果的Accuracy

+ 参数1：om模型预测结果目录
+ 参数2：imagenet纯验证集标签

```shell
python LV_Vit_postprocess.py ./result/dumpOutput_device0 ./data/val.txt
```

控制台输出如下信息

```
accuracy: 0.8317
```



### 6.2 开源精度

源代码仓公布精度

```
Model		Dataset		Accuracy
LV-Vit 		imagenet	 0.833
```



### 6.3 精度对比

将得到的om离线模型推理Accuracy与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  



## 7 性能对比

### 7.1 npu性能数据

**1. batch_size=1**

```
[e2e] throughputRate: 35.5884, latency: 1.40495e+06
[data read] throughputRate: 37.666, moduleLatency:26.5491
[preprocess] throughputRate: 37.5823, moduleLatency: 26.6802
[infer] throughputRate: 35.6308 Interface throughputRate: 37.941, moduleLatency: 27.3942
[post] throughputRate: 35.6308, moduleLatency: 28.0656
```

batch_size=1 Ascend310单卡吞吐率：37.941*4=151.764 fps



**2. batch_size=4**

```
[e2e] throughputRate: 37.4274, latency: 1.33592e+06
[data read] throughputRate: 39.6399, moduleLatency: 25.2271
[preprocess] throughputRate: 39.5442, moduleLatency: 25.2882
[infer] throughputRate: 37.4711, Interface throughputRate: 40.477, moduleLatency: 26.1715
[post] throughputRate: 9.36777, moduleLatency: 106.749
```

batch_size=4 Ascend310单卡吞吐率：40.477*4=161.908 fps



**3. batch_size=8**

```
[e2e] throughputRate: 34.8915, latency: 1.43301e+06
[data read] throughputRate: 36.8978, moduleLatency: 27.1019
[preprocess] throughputRate: 36.8307, moduleLatency: 27.1513
[infer] throughputRate: 34.9252, Interface throughputRate: 38.3992, moduleLatency: 27.4573
[post] throughputRate: 4.36564, moduleLatency: 229.062
```

batch_size=16 Ascend310单卡吞吐率：38.3992*4=153.5968 fps



**4. batch_size=16**

```
[e2e] throughputRate: 34.3406, latency: 1.456e+06
[data read] throughputRate: 36.3651, moduleLatency: 27.4989
[preprocess] throughputRate: 36.2989, moduleLatency: 27.5491
[infer] throughputRate: 34.378, Interface throughputRate: 36.9249, moduleLatency: 28.4462
[post] throughputRate: 2.14862, moduleLatency: 465.415
```

batch_size=16 Ascend310单卡吞吐率：36.9249*4=147.6996 fps



**5. batch_size=32**

```
[e2e] throughputRate: 33.136, latency: 1.50893e+06
[data read] throughputRate: 35.0612, moduleLatency: 28.5215
[preprocess] throughputRate: 34.9918, moduleLatency: 28.5781
[infer] throughputRate: 33.1637, Interface throughputRate: 36.1795, moduleLatency: 28.9776
[post] throughputRate: 1.03669, moduleLatency: 964.608
```

batch_size=16 Ascend310单卡吞吐率：36.1795*4=144.718 fps



### 7.2 npu性能优化

云盘：[model_best_bs1_sim.om](https://pan.baidu.com/s/1bMuSj4PbvuYE-pX2j_e-0Q)，提取码：ad5f

[model_best_bs16_sim.om](https://pan.baidu.com/s/11gYb6RpBbuaEL-aIql2qkg)，提取码：jiev

**1. batch_size=1**

```
[e2e] throughputRate: 40.7217, latency: 1.22785e+06
[data read] throughputRate: 43.0838, moduleLatency: 23.2106
[preprocess] throughputRate: 42.997, moduleLatency: 23.2575
[infer] throughputRate: 40.769, Interface throughputRate: 44.0188, moduleLatency: 23.7945
[post] throughputRate: 40.769, moduleLatency: 24.5285
```

batch_size=1 Ascend310单卡吞吐率：44.0188*4=176.0752 fps

**2. batch_size=16**

```
[e2e] throughputRate: 51.2825, latency: 974992
[data read] throughputRate: 54.323, moduleLatency: 18.4084
[preprocess] throughputRate: 54.1712, moduleLatency: 18.46
[infer] throughputRate: 51.3613, Interface throughputRate: 57.8179, moduleLatency: 18.6629
[post] throughputRate: 3.21005, moduleLatency: 311.521
```

batch_size=16 Ascend310单卡吞吐率：57.8179*4=231.2716 fps

### 7.3 npu性能优化前后对比

| batch size |  优化前   |  优化后  |
| :--------: | :------: | :------: |
|     1      | 151.764  | 176.0752 |
|     16     | 147.6996 | 231.2716 |



### 7.4 gpu和npu性能对比

| batch size | GPU(FPS) | NPU(FPS) |
| :--------: | -------- | -------- |
|     1      | 290.41   | 176.0752 |
|     16     | 559.35   | 231.2716 |



