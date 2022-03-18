# CGAN推理说明

## 1 模型概述

- **[论文地址](https://arxiv.org/abs/1411.1784)**
- **[代码地址](https://github.com/znxlwm/pytorch-generative-model-collections/)**

### 1.1 论文地址

[CGAN论文](https://github.com/znxlwm/pytorch-generative-model-collections/)

### 1.2 代码地址

[CGAN代码](https://github.com/znxlwm/pytorch-generative-model-collections/)

branch:master

commitid:0d183bb5ea2fbe069e1c6806c4a9a1fd8e81656f


## 2 环境说明

- 深度学习框架
- python第三方库

### 2.1 深度学习框架

```
python3.7.5
CANN 5.0.3

pytorch == 1.5.0
torchvision == 0.6.0
onnx == 1.10.2
```

### 2.2 python第三方库

```
numpy == 1.21.2
Pillow == 8.4.0
imageio == 2.9.0
scipy == 1.7.1
matplotlib==3.4.3
```

## 3 模型转换

- pth转om模型

### 3.1 pth转om模型

1.获取pth权重文件

pth权重文件随附件一起打包

2.下载CGAN推理代码

```
git clone https://gitee.com/wang-chaojiemayj/modelzoo.git
cd modelzoo
git checkout tuili
```

进入CGANt目录

```
cd ./contrib/ACL_PyTorch/Research/cv/GAN/CGAN
```

3.pth模型转onnx模型，onnx转成om模型

pth模型转onnx模型

```
python3.7 CGAN_pth2onnx.py --pth_path CGAN_G.pth --onnx_path CGAN.onnx
python3.7 -m onnxsim --input-shape="100,72" CGAN.onnx CGAN_sim.onnx
```

onnx转出om,并使用autotune优化om模型，这将耗费大量时间

```
source env.sh（注意，latest是一个软连接，请将服务器中的/usr/local/Ascend/ascend-toolkit/latest 指向5.0.3版本的CANN包）
# 生成器一次只能生成一张图，由于模型输入是两维的，不是常用的NCHW格式，input_format采用ND形式
atc --framework=5 --model=CGAN_sim.onnx --output=CGAN_bs1 --input_format=ND --output_type=FP32 --input_shape="image:100,72" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"
```

## 4 数据集预处理

- 数据集获取
- 数据预处理
- 生成数据集信息文件

### 4.1 数据集获取

本模型的输入数据由随机数以及标签生成，在CGAN_preprocess.py中会生成数据并转成二进制文件，并保存在。’./prep_dataset‘目录下。

文件结构如下

```
|CGAN--test
|     |   |--pth2om.sh
|     |   |--eval_acc_perf.sh
|     |   |--perf_t4.sh
|     |--util.py
|     |--CGAN.py
|     |--gen_dataset_info.py
|     |--env.sh
|     |--CGAN_pth2onnx.py
|     |--CGAN_preprocess.py
|     |--CGAN_postprocess.py
|     |--requirements.txt
|     |--LICENCE
|     |--modelzoo_level.txt
|     |--README.md
```


### 4.2 数据集预处理

运行CGAN_preprocess.py

```
python3.7  CGAN_preprocess.py --save_path ./prep_dataset
```

二进制文件将保存在./prep_dataset目录下

### 4.3 生成数据集信息文件

1.执行生成数据集信息脚本gen_dataset_info.py，生成数据集信息文件

```
python3.7 gen_dataset_info.py --dataset_bin ./prep_dataset --info_name CGAN_prep_bin.info --width 72 --height 100
```

## 5 离线推理

- msame概述
- 离线推理

### 5.1 msame工具概述

msame工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.1 推理msame工具用户指南 

### 5.2 离线推理

1.设置环境变量

```
source env.sh
```

2.执行离线推理

```
./msame --model "./CGAN_bs1.om" --input "./prep_dataset/input.bin" --output "./out" --outfmt BIN --loop 1
```

输出结果保存在'./out'目录下

## 6 精度对比

- 离线推理精度
- 开源精度
- 开源精度对比

### 6.1 离线推理精度统计

将msame推理获得的输出结果进行后处理，保存为图片

```
python3.7 CGAN_postprocess.py --bin_out_path ./out/20211124_090506 --save_path ./result   
```

第一个参数为msame输出目录，’/20211113_073952‘是离线推理时根据时间自动生成的目录，请根据实际情况改变，第二个参数为保存后处理产生的图片的目录。

### 6.2 开源精度

![](README.assets/CGAN_epoch050-16371406300071.png)

### 6.3 精度对比

![](README.assets/result.png)

om模型可以正常生成数字，与pth模型生成的图片大致一致。

## 7 性能对比

- NPU性能数据
- T4性能数据
- 性能对比

### 7.1 npu性能数据

1.使用msame工具执行以下指令通过纯推理获得性能数据

```
./msame --model "CGAN_bs1.om"  --output "./out" --outfmt BIN --loop 20
```

结果如下：

```
[INFO] get max dynamic batch size success
[INFO] output data success
Inference average time: 2.554200 ms
Inference average time without first time: 2.547842 ms
[INFO] destroy model input success.
```

310单卡吞吐率：1000*(1/2.547842)*4=1568fps


### 7.2 T4性能数据

在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2。
执行以下命令获取T4性能数据

```
trtexec --onnx=CGAN.onnx --fp16 --shapes=image:100,72 --threads
```

```
[11/14/2021-09:17:40] [I] GPU Compute
[11/14/2021-09:17:40] [I] min: 0.407471 ms
[11/14/2021-09:17:40] [I] max: 2.23047 ms
[11/14/2021-09:17:40] [I] mean: 0.427789 ms
[11/14/2021-09:17:40] [I] median: 0.428223 ms
[11/14/2021-09:17:40] [I] percentile: 0.4552 ms at 99%
[11/14/2021-09:17:40] [I] total compute time: 2.96629 s
```

T4单卡吞吐率：1000/(0.428223/1)=2337fps

### 7.3 性能对比

310性能：1000*(1/2.547842)*4=1568fps

T4性能：1000/(0.428223/1)=2337fps

310性能低于T4性能。

### 7.4 性能优化

autotune优化，结果如下：

![img](README.assets/wps8587.tmp-16378261403441.jpg)

优化TransData，TransPose，结果如下：

![img](README.assets/wps229E.tmp.jpg)

onnxsim优化onnx，结果如下：

![img](README.assets/wps4092.tmp.jpg)

最终经过autotune优化，优化TransData、TransPose，onnxsim优化onnx之后，最终的结果如下：

![image-20211125154623271](README.assets/image-20211125154623271.png)

最终的性能为：1000/0.065243*4=1936FPS



