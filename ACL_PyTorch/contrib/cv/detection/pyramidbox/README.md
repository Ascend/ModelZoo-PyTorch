# Pyramidbox Onnx模型端到端推理指导

- 1 模型概述
  - [1.1 论文地址]([[1803.07737\] PyramidBox: A Context-assisted Single Shot Face Detector (arxiv.org)](https://arxiv.org/abs/1803.07737))
  - [1.2 代码地址](https://gitee.com/kghhkhkljl/pyramidbox.git)
- 2 环境说明
  - [2.1 深度学习框架](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#21-深度学习框架)
  - [2.2 python第三方库](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#22-python第三方库)
- 3 模型转换
  - [3.1 pth转onnx模型](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#31-pth转onnx模型)
  - [3.2 onnx转om模型](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#32-onnx转om模型)
- 4 数据集预处理
  - [4.1 数据集获取](https://www.graviti.cn/open-datasets/WIDER_FACE)
  - [4.2 数据集预处理](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#42-数据集预处理)
  - [4.3 生成数据集信息文件](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#43-生成数据集信息文件)
- 5 离线推理
  - [5.1 benchmark工具概述](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/ResNext50#51-benchmark工具概述)
  - [5.2 离线推理](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#52-离线推理)
- 6 精度对比
  - [6.1 离线推理精度统计](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#61-离线推理精度统计)
  - [6.2 开源精度](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#62-开源精度)
  - [6.3 精度对比](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#63-精度对比)
- 7 性能对比
  - [7.1 npu性能数据](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#71-npu性能数据)
  - [7.2 T4性能数据](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#72-T4性能数据)
  - [7.3 性能对比](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/Pyramidbox#73-性能对比)

## 1 模型概述

- **论文地址**
- **代码地址**

### 1.1 论文地址

[Pyramidbox论文](https://arxiv.org/abs/1803.07737)

### 1.2 代码地址

https://gitee.com/kghhkhkljl/pyramidbox.git

## 2 环境说明

- **深度学习框架**
- **python第三方库**

### 2.1 深度学习框架

```
python3.7.5
CANN 5.0.3

pytorch >= 1.5.0
torchvision >= 0.10.0
onnx >= 1.7.0

说明：若是在conda环境下，直接采用python，不用python3.7
```

### 2.2 python第三方库

```
torch == 1.9.0
numpy == 1.20.3
Pillow == 8.2.0
opencv-python == 4.5.3.56
scipy == 1.7.1
easydict == 1.9
six == 1.16.0
pycocotools == 2.0.2
```

## 3 模型转换

- **pth转onnx模型**
- **onnx转om模型**

### 3.1 pth转onnx模型

1.拉取代码仓库 （因为使用了开源代码模块，所以需要git clone一下）

```shell
git clone https://gitee.com/kghhkhkljl/pyramidbox.git
```

克隆下来源代码之后将pr中的代码放到克隆下来的pyramidbox下面

2.下载pth权重文件
权重文件从百度网盘上获取：[pyramidbox_120000_99.02.pth_免费高速下载|百度网盘-分享无限制 (baidu.com)](https://pan.baidu.com/s/1VtzgB9srkJY4SUtVM3n8tw?_at_=1631960039538)

下载下来的权重文件也需要放在pyramidbox目录下面

3.使用pth2onnx.py进行onnx的转换

```
方法二：cd pyramidbox/test
bash pth2onnx.sh
方法二：cd pyramidbox
python3.7 pyramidbox_pth2onnx.py  ./pyramidbox_1000.onnx ./pyramidbox_120000_99.02.pth
第一个参数是onnx文件生成在当前目录的名字，第二个参数是当前目录下的权重文件
```

### 3.2 onnx转om模型

1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01

```
方法一：cd pyramidbox/test
bash onnxToom.sh 
方法二：cd pyramidbox
atc --framework=5 --model=pyramidbox_1000.onnx --input_format=NCHW --input_shape="image:1,3,1000,1000" --output=pyramidbox_1000_bs1 --log=debug --soc_version=Ascend310 --precision_mode=force_fp32

--model是onnx的文件名，--input_shape是图片的shape，--output是输出on文件的文件名
```

## 4 数据集预处理

- **数据集获取**
- **数据集预处理**
- **生成数据集信息文件**

### 4.1 数据集获取

下载WIDER_FACE数据集：

下载地址：https://www.graviti.cn/open-datasets/WIDER_FACE

可以将数据集图片放在pyramidbox目录下的images下面,images目录需要自己创建（说明：images下面是个二级目录）

```
cd pyramidbox/images
```

### 4.2 数据集预处理

1.预处理脚本pyramidbox_pth_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件

```
方法一：cd pyramidbox/test
bash pre_deal.sh
方法二：cd pyramidbox
python3.7 pyramidbox_pth_preprocess.py ./images ./data1000_1 ./data1000_2
第一个参数是预处理文件，第二个参数是数据集所在目录，第三和第四个参数是预处理后的文件名（说明：由于预处理需要进行两次图片的不同处理，所以生成的文件有两个）
```

### 4.3 生成数据集信息文件

1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```
方法一：cd pyramidbox/test
bash to_info.sh
方法二：cd pyramidbox
python3.7 get_info.py bin ./data1000_1 ./pyramidbox_pre_bin_1000_1.info 1000 1000
python3.7 get_info.py bin ./data1000_2 ./pyramidbox_pre_bin_1000_2.info 1000 1000

第一个是预处理后的数据集所在目录，第二个参数是生成的info文件名，后两个参数是图片的宽高。（说明：由于预处理会对图片进行两次处理，生成的文件有两个，所以会需要生成两个info文件）
```

## 5 离线推理

- **benchmark工具概述**
- **离线推理**

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.3推理benchmark工具用户指南 

### 5.2 离线推理

1.执行离线推理

执行前需要将benchmark.x86_64移动到执行目录下

(注：执行目录是/pyramidbox)

然后运行如下命令：

```
方法一：cd pyramidbox/test
bash infer.sh
方法二：cd pyramidbox
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./pyramidbox_1000_bs1.om -input_text_path=./pyramidbox_pre_bin_1.info -input_width=1000 -input_height=1000 -output_binary=True -useDvpp=False --precision_mode=force_fp32
./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=1 -om_path=./pyramidbox_1000_bs1.om -input_text_path=./pyramidbox_pre_bin_2.info -input_width=1000 -input_height=1000 -output_binary=True -useDvpp=False --precision_mode=force_fp32

-om_path为om所在的路径，-input_text_path为预处理后的bin文件的整个info文件，-input_width为图片的宽，-input_height为图片的高。由于预处理后的数据集有两个，所以此脚本需要运行两次，第二次运行只需要改动-device_id=1和-input_text_path为相应的info文件即可(例如：pyramidbox_pre_bin_2.info)。
```

输出结果默认保存在当前目录result/dumpOutput_device{0}以及result/dumpOutput_device{1}下，每个输入对应的输出对应2个_1.bin文件，我们只使用第一个。

2.处理目录result/dumpOutput_device{0}和result/dumpOutput_device{1}下的bin文件

将该目录下的文件分类别存放，以便于后处理

```
方法一：cd pyramidbox/test
bash convert.sh
方法二：cd pyramidbox
python3.7 convert.py ./result/dumpOutput_device0/ ./result/result1
python3.7 convert.py ./result/dumpOutput_device1/ ./result/result2
第一个参数是infer.sh脚本生成的文件，第二个参数是生成的二级目录所在的文件夹。
```



## 6 精度对比

- **离线推理精度**
- **开源精度**
- **精度对比**

### 6.1 离线推理精度统计

1.后处理

```
cd ./pyramidbox
python3.7 pyramidbox_pth_postprocess.py
```

2.进行Ascend310上精度评估

```
cd ./pyramidbox/evaluate
python3.7 evaluation.py
```

### 6.2 开源精度

pyramidbox在线推理精度：

```
Easy   Val AP: 0.958986327388428
Medium Val AP: 0.9504929578311708
Hard   Val AP: 0.907248372271328
```

### 6.3 精度对比

```
Easy   Val AP: 0.9628280209085509
Medium Val AP: 0.9538134269337523
Hard   Val AP: 0.8798007442124222
```

### 6.3 精度对比

由于源码没有固定住shape，所以精度会有损失，因此和同一分辨率下的在线推理进行对比。对比方式：三个尺度求和取平均。

## 7 性能对比

- **npu性能数据**
- **T4性能数据**
- **性能对比**

### 7.1 npu性能数据

1.benchmark工具在整个数据集上推理获得性能数据
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：

```
[e2e] throughputRate: 0.609815, latency: 5.29013e+06
[data read] throughputRate: 0.635586, moduleLatency: 1573.35
[preprocess] throughputRate: 0.61536, moduleLatency: 1625.07
[infer] throughputRate: 0.6099, Interface throughputRate: 0.620281, moduleLatency: 1638.44
[post] throughputRate: 0.6099, moduleLatency: 1639.61
```

Interface throughputRate: 0.620281，0.620281x4=2.48既是batch1 310单卡吞吐率



说明：由于bs2以上会导致爆显存，所以测不了性能，此处只测了bs1。

![1633688929248](C:\Users\Eiven\AppData\Roaming\Typora\typora-user-images\1633688929248.png)

### 7.2 T4性能数据

batch1 t4单卡吞吐率的计算方法是通过计算平均每张图片的耗时t，然后用1/t即是batch1 t4的单卡吞吐率。此处的t=1.560808，所以吞吐率为0.6407

### 7.3 性能对比

batch1：0.620281x4=2.48>0.6407