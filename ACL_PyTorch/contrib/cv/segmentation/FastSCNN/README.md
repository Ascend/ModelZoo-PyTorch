# FastSCNN推理说明

## 1 模型概述

- **[论文地址](https://arxiv.org/abs/1902.04502)**
- **[代码地址](https://gitee.com/wang-chaojiemayj/modelzoo/tree/master/contrib/PyTorch/Research/cv/image_segmentation/FastSCNN)**

### 1.1 论文地址

[FastSCNN论文](https://arxiv.org/abs/1902.04502)

### 1.2 代码地址

[FascSCNN代码](https://gitee.com/wang-chaojiemayj/modelzoo/tree/master/contrib/PyTorch/Research/cv/image_segmentation/FastSCNN)

branch:master

commitid:e86409484cf89467a569be43acee1b3f06b92305 


## 2 环境说明

- 深度学习框架
- python第三方库

### 2.1 深度学习框架

```
python3.7.5
CANN 5.0.2

pytorch >= 1.5.0
torchvision >= 0.6.0
onnx == 1.7.0
onnx-simplifier == 0.3.6
```

### 2.2 python第三方库

```
numpy == 1.21.1
Pillow == 8.3.0
opencv-python == 4.5.2.54
```

## 3 模型转换

- pth转om模型

### 3.1 pth转om模型

1.获取pth权重文件

获取权重文件方法，可从Ascend modelzoo FastSCNN_ACL_Pytorch 模型压缩包获取

md5sum:efc7247270298f3f57e88375011b52ee

2.FastSCNN模型已经上传至代码仓，使用git工具获取FastSCNN模型代码
使用gitclone获取模型训练的代码，切换到tuili分支。

```
git clone https://gitee.com/wang-chaojiemayj/modelzoo.git
cd modelzoo
git checkout tuili
```

进入FastSCNN目录

```
cd ./contrib/ACL_PyTorch/Research/cv/segmentation/FastSCNN
```

使用gitclone下载模型代码

```
git clone https://github.com/LikeLy-Journey/SegmenTron
```

由于onnx不支持AdaptiveAvgPool算子，需要使用module.patch修改module.py。
将FastSCNN目录下的module.patch放到FastSCNN/SegmenTron目录下
执行

```
cd ./SegmenTron
git apply module.patch
cd ..
```

3.执行pth2om脚本，生成om模型文件

ascend-toolkit版本：5.0.2
```
bs1：
python3.7 Fast_SCNN_pth2onnx.py --pth_path best_model.pth --onnx_name fast_scnn_bs1 --batch_size 1
bs16：
python3.7 Fast_SCNN_pth2onnx.py --pth_path best_model.pth --onnx_name fast_scnn_bs16 --batch_size 16
 **bs4：bs16无法导出时使用
python3.7 Fast_SCNN_pth2onnx.py --pth_path best_model.pth --onnx_name fast_scnn_bs4 --batch_size 4** 
``` 
参数说明：
--pth_path：pth权重文件的路径，可自行设置，默认值为best_model.pth；
--onnx_name：需要转出的onnx模型的名称，可自行设置，默认值为fast_scnn_bs1（由于本模型不支持动态batch，推荐在模型名后加后缀，如‘_bs1’，用以区分不同batch_size的onnx模型）;
--batch_size：导出的onnx模型的batch_size，可自行设置，默认值为1。

onnx转出om

bs1：
```
source env.sh（注意，latest是一个软连接，请将服务器中的/usr/local/Ascend/ascend-toolkit/latest 指向5.0.2版本的CANN包）
atc --framework=5 --model=fast_scnn_bs1.onnx --output=fast_scnn_bs1  --output_type=FP16 --input_format=NCHW --input_shape="image:1,3,1024,2048" --log=debug --soc_version=Ascend310
```
bs16：
```
source env.sh
atc --framework=5 --model=fast_scnn_bs16.onnx --output=fast_scnn_bs16  --output_type=FP16 --input_format=NCHW --input_shape="image:16,3,1024,2048" --log=debug --soc_version=Ascend310
```
bs4:（bs16无法离线推理时使用）
```
source env.sh
atc --framework=5 --model=fast_scnn_bs4.onnx --output=fast_scnn_bs4  --output_type=FP16 --input_format=NCHW --input_shape="image:4,3,1024,2048" --log=debug --soc_version=Ascend310
```


## 4 数据集预处理

- 数据集获取
- 数据预处理
- 生成数据集信息文件

### 4.1 数据集获取

本模型支持cityscapes leftImg8bit的500张验证集。用户需要下载[leftImg8bit_trainvaltest.zip](http://www.cityscapes-dataset.com/downloads)和[gtFine_trainvaltest.zip](http://www.cityscapes-dataset.com/downloads)数据集，解压，将两个数据集放在/opt/npu/datasets/cityscapes/目录下。推荐使用软连接，可以节省时间，数据集目录如下。

```
|opt--npu--datasets
|          |-- cityscapes
|          |   |-- gtFine
|          |   |   |-- test
|          |   |   |-- train
|          |   |   |-- val
|          |   |-- leftImg8bit
|          |       |-- test
|          |       |-- train
|          |       |-- val
```


### 4.2 数据集预处理

在modelzoo/contrib/ACL_PyTorch/Research /cv/segmentation/FastSCNN目录创建软连接

```
ln -s /opt/npu/datasets datasets
```

运行Fast_SCNN_preprocess.py

```
python3.7  Fast_SCNN_preprocess.py
```

数据预处理的结果会保存在/opt/npu/prep_datset
预处理之后的二进制文件目录如下：
/opt/npu/prep_dataset/datasets/leftImg8bit/
/opt/npu/prep_dataset/datasets/gtFine/
在modelzoo/contrib/ACL_PyTorch/Research/cv/segmentation/FastSCNN目录下创建软连接

```
ln -s /opt/npu/prep_dataset prep_dataset
```

### 4.3 生成数据集信息文件

1.生成数据集信息文件脚本gen_dataset_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```
python3.7 gen_dataset_info.py 
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
bs1：
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=fast_scnn_bs1.om -input_text_path=./fast_scnn_prep_bin.info -input_width=2048 -input_height=1024 -output_binary=True -useDvpp=False./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=4 -om_path=fast_scnn_bs4.om -input_text_path=./fast_scnn_prep_bin.info -input_width=2048 -input_height=1024 -output_binary=True -useDvpp=False
```
bs16：
```
./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=16 -om_path=./fast_scnn_bs16.om -input_text_path=./fast_scnn_prep_bin.info -input_width=2048 -input_height=1024 -output_binary=True -useDvpp=False
```
bs4:（bs16无法离线推理时使用）
```
./benchmark.x86_64 -model_type=vision -device_id=2 -batch_size=4 -om_path=fast_scnn_bs4.om -input_text_path=fast_scnn_prep_bin.info -input_width=2048 -input_height=1024 -output_binary=True -useDvpp=False
```
```
参数说明：
需要更改的参数
-device_id：使用的Ascend310处理器的卡号，可选0、1、2、3，尽量选择不同的卡号进行推理，若-device_id=0，离线推理结果会保存在./result/dumpOut
put_device0中，device0中的0代表卡号是0；
-batch_size：om模型的batch_size;
-om_path:  需要进行离线推理的om模型的路径；
不需要更改的参数：
-input_text_path：om模型的二进制输入图片的路径信息文件的路径；
-input_width：输入图片的宽度，FastSCNN模型是2048；
-input_heigh：输入图片的高度，FastSCNN模型是1024；
-output_binary：benchmark的输出是二进制文件还是txt文件，True代表输出为二进制文件；
-useDvpp：是否使用Dvpp工具，FastSCNN模型不使用Dvpp工具，设置为False；
```
## 6 精度对比

- 离线推理精度
- 开源精度
- 开源精度对比

### 6.1 离线推理精度统计

后处理统计mIoU

调用cityscapes_acc_eval.py脚本推理结果与label比对，获取pixAcc和mIoU数据，结果保存在fast_scnn_bs1.log和fast_scnn_bs4.log

```
python3.7 cityscapes_acc_eval.py result/dumpOutput_device0/ ./out >fast_scnn_bs1.log
python3.7 cityscapes_acc_eval.py result/dumpOutput_device1/ ./out >fast_scnn_bs4.log
```

第一个为benchmark输出目录，第二个为输出重定向文件名

```
pixAcc：94.29% mIoU：64.43
```

经过对bs1与bs4的om测试，本模型batch1的精度与batch4的精度一致，精度数据如上

### 6.2 开源精度

pth精度

```
Model      pixAcc     mIoU
FastSCNN   93.877%   64.46
```

### 6.3 精度对比

将得到的om模型离线推理精度与pth精度作比较，精度下降不超过0.5%，故精度达标

## 7 性能对比

- NPU性能数据
- T4性能数据
- 性能对比

### 7.1 npu性能数据

1.benchmark工具在整个数据集上推理获得性能数据。
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：

```
[e2e] throughputRate: 0.99593, latency: 502043[data read] throughputRate: 1.62003, moduleLatency: 617.273[preprocess] throughputRate: 1.20942, moduleLatency: 826.844[inference] throughputRate: 1.02697, Interface throughputRate: 5.5718, moduleLatency: 973.739[postprocess] throughputRate: 0.999452, moduleLatency: 1000.55
```

Interface throughputRate: 5.5718，5.5718x4=22.286既是batch1 310单卡吞吐率

batch4的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_4_device_1.txt：

```
[e2e] throughputRate: 0.429745, latency: 1.16348e+06[data read] throughputRate: 0.673692, moduleLatency: 1484.36[preprocess] throughputRate: 0.525523, moduleLatency: 1902.86[inference] throughputRate: 0.477698, Interface throughputRate: 5.59273, moduleLatency: 2100.2[postprocess] throughputRate: 0.107216, moduleLatency: 9327
```

Interface throughputRate: 5.59273，5.59273x4=22.37092既是batch16 310单卡吞吐率


### 7.2 T4性能数据

在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2
batch1性能：

```
trtexec --onnx=fast_scnn_bs1.onnx --fp16 --shapes=image:1x3x1024x2048 --threads
```

```
[07/20/2021-01:49:12] [I] GPU Compute[07/20/2021-01:49:12] [I] min: 6.1626 ms[07/20/2021-01:49:12] [I] max: 6.18018 ms[07/20/2021-01:49:12] [I] mean: 6.17022 ms[07/20/2021-01:49:12] [I] median: 6.17062 ms[07/20/2021-01:49:12] [I] percentile: 6.18018 ms at 99%[07/20/2021-01:49:12] [I] total compute time: 0.265319 s
```

batch1 t4单卡吞吐率：1000/(6.17022/1)=162.068fps

batch4性能：

```
trtexec --onnx=fast_scnn_bs4.onnx --fp16 --shapes=image:4x3x1024x2048 --threads
```

```
[08/25/2021-05:18:21] [I] GPU Compute[08/25/2021-05:18:21] [I] min: 23.7666 ms[08/25/2021-05:18:21] [I] max: 24.3643 ms[08/25/2021-05:18:21] [I] mean: 24.0295 ms[08/25/2021-05:18:21] [I] median: 23.9731 ms[08/25/2021-05:18:21] [I] percentile: 24.3643 ms at 99%[08/25/2021-05:18:21] [I] total compute time: 0.288354 s
```

batch4 t4单卡吞吐率：1000/(24.0295/4)=166.46fps

### 7.3 性能对比

batch1：5.5718x4 < 1000/(6.17022/1)
batch2：5.59273x4 <1000/(24.0295/4)
310单个device的吞吐率乘4即单卡吞吐率比T4单卡的吞吐率小，故310性能低于T4性能，性能不达标。
对于batch1与batch4，310性能均低于T4性能，该模型放在contrib/ACL_PyTorch/Research目录下。
**性能优化：**

测试版本：CANN 5.0.2

目前可行的解决方案有三个:

（1）优化TransData，修改five_2_four.py和four_2_five.py

（2）输出节点由float32改为float16

（3）模型中Resize节点的mode由双线性为最近邻

具体优化方法如下：

（1）修改five_2_four.py和four_2_five.py从profiling数据的op_statistic_0_1.csv看出影响性能的是TransData，ResizeBilinearV2D，AvgPoolV2算子。从op_summary_0_1.csv可以看出单个TransData的aicore耗时，确定可以可以优化。

```
five_2_four.py：9928    修改如下：  elif dst_format.lower() == "nchw" and dst_shape in [[2560, 512, 4, 26], [2560, 512, 1, 26], [2560, 256, 8, 25],[16, 240, 7, 7], [16, 120, 14, 14], [1, 128, 32, 64], [1, 19, 1024, 2048], [2, 128, 32, 64], [2, 19, 1024, 2048]]:
```

```
 four_2_five.py：1219   修改如下：   if src_format.upper() == "NCHW" and shape_input in [[16, 240, 7, 7], [16, 120, 14, 14],[1, 1, 1024, 2048, 16], [1, 8, 32, 64, 16], [2, 1, 1024, 2048, 16], [2, 8, 32, 64, 16]] and dtype_input == "float16":
```

(2)指定输出为fp16：

```
atc --framework=5 --model=fast_scnn_bs1_sim.onnx --output=fast_scnn_bs1  --output_type=FP16 --input_format=NCHW --input_shape="image:1,3,1024,2048" --log=debug --soc_version=Ascend310python3.7.5 -m onnxsim  --input-shape="2,3,1024,2048" fast_scnn_bs2.onnx fast_scnn_bs2_sim.onnx --skip-optimizationatc --framework=5 --model=fast_scnn_bs2_sim.onnx --output=fast_scnn_bs2  --output_type=FP16 --input_format=NCHW --input_shape="image:2,3,1024,2048" --log=debug --soc_version=Ascend310
```

(3)模型中Resize节点的mode由双线性为最近邻

```
newnode229 = onnx.helper.make_node(    'Resize',    name='Resize_229',    inputs=['549', '560', '561', '559'],    outputs=['562'],    coordinate_transformation_mode='align_corners',    cubic_coeff_a=-0.75,    mode='nearest',    nearest_mode='floor')newnode245 = onnx.helper.make_node(    'Resize',    name='Resize_245',    inputs=['566', '577', '578', '576'],    outputs=['579'],    coordinate_transformation_mode='align_corners',    cubic_coeff_a=-0.75,    mode='nearest',    nearest_mode='floor')graph.node.remove(model.graph.node[126])graph.node.insert(126,newnode126)graph.node.remove(model.graph.node[144])graph.node.insert(144,newnode144)graph.node.remove(model.graph.node[162])graph.node.insert(162,newnode162)graph.node.remove(model.graph.node[180])graph.node.insert(180,newnode180)graph.node.remove(model.graph.node[185])graph.node.insert(185,newnode185)graph.node.remove(model.graph.node[213])graph.node.insert(213,newnode213)graph.node.remove(model.graph.node[229])graph.node.insert(229,newnode229)graph.node.remove(model.graph.node[245])graph.node.insert(245,newnode245)onnx.checker.check_model(model)onnx.save(model, 'bs1_resized.onnx')
```

（4）性能、精度统计

| 方法                                              | 精度                        | 性能                        |
| ------------------------------------------------- | --------------------------- | --------------------------- |
| 未优化                                            |                             |                             |
| 优化TransData，修改five_2_four.py和four_2_five.py | bs1:mIoU64.46;bs2:mIoU64.46 | bs1:4.135fps;bs2:6.265fps   |
| 输出节点由float32改为float16                      | bs1:mIoU64.43;bs2:mIoU64.43 | bs1:22.518fps;bs2:22.694fps |
| 模型中Resize节点的mode由双线性为最近邻            | bs1:mIoU60.41;bs1:mIoU60.41 | bs1:7.747fps;bs2:14.046fps  |

8.本模型经过指定输出结点为fp16后，精度为64.43，pth精度为64.46，精度达标；性能提高到22fps左右，故本次pr提交的模型输出为结点fp16。

