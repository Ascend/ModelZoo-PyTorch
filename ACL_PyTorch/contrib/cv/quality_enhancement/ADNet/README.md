# ADNet推理说明

## 1 模型概述

- **[论文地址](https://www.sciencedirect.com/science/article/pii/S0893608019304241)**
- **[代码地址](https://github.com/hellloxiaotian/ADNet)**

### 1.1 论文地址

[ADNet论文](https://www.sciencedirect.com/science/article/pii/S0893608019304241)

### 1.2 代码地址

[ADNet代码](https://github.com/hellloxiaotian/ADNet)

branch:master

commitid:commit 997df8f0cd5cebe2d26a1468c866dd927512686f


## 2 环境说明

- 深度学习框架
- python第三方库

### 2.1 深度学习框架

```
python3.7.5
CANN 5.0.2

pytorch == 1.5.0
torchvision == 0.5.0
onnx == 1.7.0
onnx-simplifier == 0.3.6
```

### 2.2 python第三方库

```
numpy == 1.21.2
Pillow == 8.3.0
opencv-python == 4.5.3.56
scikit-image==0.16.2
```

## 3 模型转换

- pth转om模型

### 3.1 pth转om模型

1.获取pth权重文件

[pth权重文件](https://github.com/hellloxiaotian/ADNet/blob/master/gray/g25/model_70.pth
) md5sum:7a93fb1f437cbce0fd235daaa7b9cffd 

2.下载ADNet推理代码

```
git clone https://gitee.com/wang-chaojiemayj/modelzoo.git
cd modelzoo
git checkout tuili
```
进入ADNet目录
```
cd ./contrib/ACL_PyTorch/Research/cv/quality_enhancement/ADnet
```
3.pth模型转onnx模型，onnx转成om模型

pth模型转onnx模型
```
python3.7.5 ADNet_pth2onnx.py model_70.pth ADNet.onnx
```
onnx转出om
```
source env.sh（注意，latest是一个软连接，请将服务器中的/usr/local/Ascend/ascend-toolkit/latest 指向5.0.2版本的CANN包）
# bs1
atc --framework=5 --model=ADNet.onnx --output=ADNet_bs1 --input_format=NCHW --input_shape="image:1,1,321,481" --log=debug --soc_version=Ascend310 
#bs16
atc --framework=5 --model=ADNet.onnx --output=ADNet_bs16 --input_format=NCHW --input_shape="image:16,1,321,481" --log=debug --soc_version=Ascend310 
```

## 4 数据集预处理

- 数据集获取
- 数据预处理
- 生成数据集信息文件

### 4.1 数据集获取

本模型支持BSD68数据集共68张数据集，可从百度云盘下载

链接：https://pan.baidu.com/s/1XiePOuutbAuKRRTV949FlQ 
提取码：0315

文件结构如下

```
|ADNet--test
|     |  |--pth2om.sh
|     |  |--perf_t4.sh
|     |  |--parse.py
|     |  |--eval_acc_perf.sh
|     |--datset
|     |  |--BSD68
|     |--prep_dataset
|     |  |--ISoure
|     |  |--INoisy
|     |--util.py
|     |--requirements.tx
|     |--models.py
|     |--gen_dataset_info.py
|     |--env.sh
|     |--ADNet_pth2onnx.py
|     |--ADNet_preprocess.py
|     |--ADNet_postprocess.py
```


### 4.2 数据集预处理

运行ADNet_preprocess.py
```
python3.7.5  ADNet_preprocess.py ./dataset/BSD68 ./prep_dataset
```
二进制文件将保存在./prep_dataset目录下

### 4.3 生成数据集信息文件

1.执行生成数据集信息脚本gen_dataset_info.py，生成数据集信息文件

```
python3.7.5 gen_dataset_info.py ./prep_dataset/INoisy ADNet_prep_bin.info 481 321  
```

## 5 离线推理

- benchmark工具概述
- 离线推理

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.1 推理benchmark工具用户指南 01

### 5.2 离线推理

1.设置环境变量

```
source env.sh
```

2.执行离线推理

```
bs1:
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./ADNet_bs1.om -input_text_path=./ADNet_prep_bin.info -input_width=481 -input_height=321 -output_binary=True -useDvpp=False
bs16:
./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=16 -om_path=./ADNet_bs16.om -input_text_path=./ADNet_prep_bin.info -input_width=481 -input_height=321 -output_binary=True -useDvpp=False
```

输出结果分别保存在当前目录result/dumpOutput_device0和result/dumpOutput_device1中，模型的输出有三个，其中需要的是名为output1的输出，shape为(1,19,1024,2048)(NCHW)，数据类型为FP16，每个输入对应的输出对应三个_x.bin(x代表1,2,3）文件。

## 6 精度对比

- 离线推理精度
- 开源精度
- 开源精度对比

### 6.1 离线推理精度统计

后处理统计PSNR精度

调用ADNet_postprocess.py脚本推理结果与label比对，获取PSNRj精度数据，结果保存在ADNet_bs1.log和ADNet_bs4.log

```
python3.7.5 -u ADNet_postprocess.py ./result/dumpOutput_device0 ./prep_dataset/ISoure ./out >ADNet_bs1.log
python3.7.5 -u ADNet_postprocess.py ./result/dumpOutput_device1 ./prep_dataset/ISoure ./out >ADNet_bs16.log
```

第一个为benchmark输出目录，第二个标签目录，第三个为重定向输出目录

```
PSNR：29.68
```

经过对bs1与bs6的om测试，本模型batch1的精度与batch4的精度一致，精度数据如上
### 6.2 开源精度

pth精度

```
Model   论文    开源pth文件
ADNet   29.27     29.25
```

### 6.3 精度对比

将得到的om模型离线推理精度与pth精度作比较，om模型精度高于pth模型精度，精度达标。

## 7 性能对比

- NPU性能数据
- T4性能数据
- 性能对比

### 7.1 npu性能数据

1.benchmark工具在整个数据集上推理获得性能数据。
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：

```
-----------------Performance Summary------------------
[e2e] throughputRate: 21.1584, latency: 3213.85
[data read] throughputRate: 2267.5, moduleLatency: 0.441015
[preprocess] throughputRate: 613.431, moduleLatency: 1.63018
[inference] throughputRate: 33.8299, Interface throughputRate: 35.7852, moduleLatency: 29.1051
[postprocess] throughputRate: 34.309, moduleLatency: 29.1469

-----------------------------------------------------------
```

Interface throughputRate: 35.7852，35.7852x4=143.1408即是batch1 310单卡吞吐率

batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：

```
-----------------Performance Summary------------------
[e2e] throughputRate: 19.8971, latency: 3417.58
[data read] throughputRate: 2382.7, moduleLatency: 0.419691
[preprocess] throughputRate: 405.505, moduleLatency: 2.46606
[inference] throughputRate: 27.4387, Interface throughputRate: 29.3584, moduleLatency: 35.5952
[postprocess] throughputRate: 2.40737, moduleLatency: 415.392

-----------------------------------------------------------
```

Interface throughputRate: 29.3584，29.3584x4=117.4336即是batch16 310单卡吞吐率


### 7.2 T4性能数据

在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2
batch1性能：

```
trtexec --onnx=ADNet.onnx --fp16 --shapes=image:1x1x321x481 --threads
```

```
[09/27/2021-11:20:55] [I] GPU Compute
[09/27/2021-11:20:55] [I] min: 7.94897 ms
[09/27/2021-11:20:55] [I] max: 12.2207 ms
[09/27/2021-11:20:55] [I] mean: 8.39391 ms
[09/27/2021-11:20:55] [I] median: 8.30371 ms
[09/27/2021-11:20:55] [I] percentile: 11.1882 ms at 99%
[09/27/2021-11:20:55] [I] total compute time: 3.01341 s
```
batch1 t4单卡吞吐率：1000/(8.39391/1)=119.134fps

batch16性能：

```
trtexec --onnx=ADNet.onnx --fp16 --shapes=image:16x1x321x481 --threads
```

```
[09/27/2021-11:28:53] [I] GPU Compute
[09/27/2021-11:28:53] [I] min: 125.424 ms
[09/27/2021-11:28:53] [I] max: 138.322 ms
[09/27/2021-11:28:53] [I] mean: 128.206 ms
[09/27/2021-11:28:53] [I] median: 126.907 ms
[09/27/2021-11:28:53] [I] percentile: 138.322 ms at 99%
[09/27/2021-11:28:53] [I] total compute time: 3.33335 s
```

batch4 t4单卡吞吐率：1000/(128.206/16)=124.799fps

### 7.3 性能对比

batch1：35.7852x4  > 1000/(8.39391/1)
batch16：29.3584x4  < 000/(128.206/16)
310单个device的吞吐率乘4即单卡吞吐率与比T4单卡相比，batch1的性能：310高于T4，batch16的性能：310是T4的0.954倍，略低于T4。该模型放在contrib/ACL_PyTorch/Research目录下。

310与T4同时使用纯推理对batch16进行性能测试，310性能如下：

```
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 36.1295samples/s, ave_latency: 27.6788ms
----------------------------------------------------------------
```

batch16纯推理的性能为：36.1295x4=144.518fps

144.518>124.799，在纯推理测试性能的情况下，310性能优于T4性能。
