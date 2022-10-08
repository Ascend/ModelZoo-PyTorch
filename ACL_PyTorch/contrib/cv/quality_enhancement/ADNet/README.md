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
python3.7 ADNet_pth2onnx.py model_70.pth ADNet.onnx
```
onnx转出om
```
source env.sh（注意，latest是一个软连接，请将服务器中的/usr/local/Ascend/ascend-toolkit/latest 指向5.0.2版本的CANN包）
# bs16
atc --framework=5 --model=ADNet.onnx --output=ADNet_bs16 --input_format=NCHW --input_shape="image:16,1,321,481" --log=debug --soc_version=Ascend${chip_name}
```
- model：为ONNX模型文件。

- framework：5代表ONNX模型。

- output：输出的OM模型。

- input_format：输入数据的格式。

- input_shape：输入数据的shape。

- log：日志级别。

- soc_version：处理器型号。

  运行成功后生成ADNet_bs16.om模型文件。

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
|     |--ADNet_pth2onnx.py
|     |--ADNet_preprocess.py
|     |--ADNet_postprocess.py
```


### 4.2 数据集预处理

运行ADNet_preprocess.py
```
python3.7  ADNet_preprocess.py ./dataset/BSD68 ./prep_dataset
```
二进制文件将保存在./prep_dataset目录下

### 4.3 生成数据集信息文件

1.执行生成数据集信息脚本gen_dataset_info.py，生成数据集信息文件

```
python3.7 gen_dataset_info.py ./prep_dataset/INoisy ADNet_prep_bin.info 481 321  
```

## 5 离线推理

- benchmark工具概述
- 离线推理

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.1 推理benchmark工具用户指南 01

### 5.2 离线推理

1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.执行离线推理

```
bs16:
./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=16 -om_path=./ADNet_bs16.om -input_text_path=./ADNet_prep_bin.info -input_width=481 -input_height=321 -output_binary=True -useDvpp=False
```

输出结果保存在目录result中，模型的输出有三个，其中需要的是名为output1的输出，shape为(1,19,1024,2048)(NCHW)，数据类型为FP16，每个输入对应的输出对应三个_x.bin(x代表1,2,3）文件。

## 6 精度对比

- 离线推理精度
- 开源精度
- 开源精度对比

### 6.1 离线推理精度统计

调用ADNet_postprocess.py脚本推理结果与label比对，获取PSNR精度数据，结果直接输出

```
python3.7 ADNet_postprocess.py ./result/dumpOutput_device0/ ./prep_dataset/ISoure/
```

第一个为benchmark输出目录，第二个标签目录，第三个为重定向输出目录

```
average psnr_val: 29.245188707184248
```

经过对比所有bs的om测试，在310上本模型batch1的精度与所有bs的精度一致，精度数据如上

```
average psnr_val: 29.245187698134
```

经过对比所有bs的om测试，在310P上本模型batch1的精度与所有bs的精度一致，精度数据如上

### 6.2 开源精度

pth精度

```
Model   论文    开源pth文件
ADNet   29.27     29.25
```

### 6.3 精度对比

将得到的om模型在 310 离线推理精度与pth精度作比较，om模型精度基本等于pth模型精度，精度达标。

将得到的om模型在 310 和 310P 上分别验证推理精度，310P 上bachsize1和性能最优batchsize的精度一致且大于310（或论文精度）的99%

## 7 性能对比

- NPU性能数据（310）
- NPU性能数据（310P）
- 性能对比

### 7.1 npu性能数据（310）

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

### 7.2 npu性能数据（310P）

1.benchmark工具在整个数据集上推理获得性能数据。
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：

```
-----------------Performance Summary------------------
[e2e] throughputRate: 49.0305, latency: 1407.29
[data read] throughputRate: 170.656, moduleLatency: 5.85975
[preprocess] throughputRate: 162.816, moduleLatency: 6.1419
[inference] throughputRate: 145.715, Interface throughputRate: 172.516, moduleLatency: 6.58854
[postprocess] throughputRate: 89.2291, moduleLatency: 11.2071

-----------------------------------------------------------
```

Interface throughputRate: 172.516 即是batch1 310P单卡吞吐率

batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：

```
-----------------Performance Summary------------------
[e2e] throughputRate: 48.9669, latency: 1409.11
[data read] throughputRate: 2623.08, moduleLatency: 0.381232
[preprocess] throughputRate: 713.068, moduleLatency: 1.40239
[inference] throughputRate: 136.066, Interface throughputRate: 182.893, moduleLatency: 6.66077
[postprocess] throughputRate: 11.4454, moduleLatency: 87.3713

-----------------------------------------------------------
```

Interface throughputRate: 182.893 即是batch16 310P单卡吞吐率

### 7.4 性能对比

#### 310P 与 310 性能对比

310P best 为bs16，吞吐率为 216.098 fps

310 best 为bs1， 吞吐率为 142.394fps

310P/310 = 1.495322837 > 1.2

即310P 对比 310 性能达标
