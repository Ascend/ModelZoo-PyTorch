## GAN Onnx模型PyTorch端到端推理指导

### 1 模型概述

#### 1.1 论文地址

[GAN论文](https://arxiv.org/abs/1406.2661)



#### 1.2 代码地址

[GAN代码](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py)



### 2 环境说明

#### 2.1 深度学习框架

```
CANN 5.0.2
pytorch = 1.6.0
torchvision = 0.6.0
onnx = 1.8.0
```



#### 2.2 python第三方库

```
numpy == 1.21.1
```

**说明：**

> X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
> Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装



### 3 模型转换

#### 3.1 pth转onnx模型

1. 下载pth权重文件

   [GAN预训练pth权重文件](https://wws.lanzoui.com/ikXFJvljkab)  
   解压至当前工作目录


   

2. 编写pth2onnx脚本GAN_pth2onnx.py

   
3. 执行pth2onnx脚本，生成onnx模型文件

   ```py
   python3.7 GAN_pth2onnx.py --input_file=generator_8p_0.0008_128.pth --output_file=GAN.onnx
   ```



#### 3.2 onnx转om模型

1. 设置环境变量

   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

2. 使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01

   ```
   atc --model=GAN.onnx --framework=5 --output=GAN_bs1 --input_format=NCHW --input_shape="Z:1,100" --log=error --soc_version=Ascend310
   ```  
   通过调节input_shape的第一个参数为16，64可以生成bs为16，64的om文件

### 4 数据准备

#### 4.1 生成输入数据并保存为.bin文件
由于源代码中未提供测试数据，这里调用GAN_testdata.py来生成测试数据，保存在/vectors文件夹下
   ```
   python3.7 GAN_testdata.py --online_path=images --offline_path=vectors --pth_path=generator_8p_0.0008_128.pth --iters 100 --batch_size 64
   ```



### 5 离线推理

####  5.1 msame工具概述

msame工具为华为自研的模型推理工具，输入.om模型和模型所需要的输入bin文件，输出模型的输出数据文件，支持多次推理（指对同一输入数据进行推理）。

模型必须是通过atc工具转换的om模型，输入bin文件需要符合模型的输入要求（支持模型多输入）。



####  5.2 离线推理

```
./msame --model "GAN_bs64.om"  --input "./vectors" --output "out" 
```

输出结果默认保存在当前目录out/下，为保存模型输入tensor数据的txt文件



### 6 精度对比

#### 6.1 离线推理精度

调用GAN_txt2jpg.py来进行后处理

```python
python3.7 GAN_txt2jpg.py --txt_path=out --infer_results_path=genimg
```

详细的结果输出在genimg文件夹中，可以和images文件夹下的在线推理结果做对比，看得出离线推理生成的图片质量更好


#### 6.2 精度对比

源码中未有精度对比部分，这里以两种不同的方式对同一输入的输出结果对比为准。



### 7 性能对比

#### 7.1 npu性能数据
运行下列命令

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --model=GAN.onnx --framework=5 --output=GAN_bs1 --input_format=NCHW --input_shape="Z:1,100" --log=error --soc_version=Ascend310
```

得到size为1*100的om模型



**msame工具在整个数据集上推理获得性能数据**

batch1的性能

```
Inference average time : 0.43 ms
Inference average time without first time: 0.43 ms
```

Inference average time : 0.43 ms，1000/(0.43/4)既是batch1 310单卡吞吐率 

bs1 310单卡吞吐率：9302.326fps

batch16的性能

```
Inference average time : 0.47 ms
Inference average time without first time: 0.47 ms
```

Inference average time : 0.51 ms，1000/(0.45/64)既是batch16 310单卡吞吐率 

bs16 310单卡吞吐率：136170.213fps

#### 7.2 T4性能数据

在装有T4卡的服务器上使用TensorRT测试gpu性能，测试过程请确保卡没有运行其他任务。

batch1性能：

```
./trtexec --onnx=GAN.onnx --fp16 --shapes=image:1x100
```

gpu T4是4个device并行执行的结果，mean是时延（tensorrt的时延是batch个数据的推理时间），即吞吐率的倒数乘以batch

```
[11/11/2021-13:11:22] [I] min: 0.048584 ms
[11/11/2021-13:11:22] [I] max: 4.11572 ms
[11/11/2021-13:11:22] [I] median: 0.0817871 ms
[11/11/2021-13:11:22] [I] GPU Compute
[11/11/2021-13:11:22] [I] min: 0.048584 ms
[11/11/2021-13:11:22] [I] max: 4.13281 ms
[11/11/2021-13:11:22] [I] mean: 0.0826078 ms
[11/11/2021-13:11:22] [I] median: 0.0856934 ms
[11/11/2021-13:11:22] [I] percentile: 0.118164 ms at 99%
[11/11/2021-13:11:22] [I] total compute time: 1.82233 s
```

batch1 t4单卡吞吐率：1000/(0.0826078/1)=12105.394fps 

batch16性能：
```
./trtexec --onnx=GAN.onnx --fp16 --shapes=image:1x100
```

```
[11/11/2021-13:18:27] [I] min: 0.0540771 ms
[11/11/2021-13:18:27] [I] max: 5.42334 ms
[11/11/2021-13:18:27] [I] median: 0.0800781 ms
[11/11/2021-13:18:27] [I] GPU Compute
[11/11/2021-13:18:27] [I] min: 0.0499878 ms
[11/11/2021-13:18:27] [I] max: 5.44055 ms
[11/11/2021-13:18:27] [I] mean: 0.0887248 ms
[11/11/2021-13:18:27] [I] median: 0.0830078 ms
[11/11/2021-13:18:27] [I] percentile: 0.145508 ms at 99%
[11/11/2021-13:18:27] [I] total compute time: 1.91122 s
```

batch16 t4单卡吞吐率：1000/(0.0887248/1)=180332.895fps

#### 7.3 性能对比

batch1：8510.638fps > 12105.394×0.5 fps 

batch16：125490.196fps > 180332.895×0.5 fps

性能达到基准线一半