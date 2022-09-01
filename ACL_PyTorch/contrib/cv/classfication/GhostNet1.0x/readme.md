# GhostNet1.0x模型-推理指导
-   [概述](#jump1)
-   [推理环境准备](#jump2)
-   [快速上手](#jump3)
    - [获取源码](#jump4)
    - [准备数据集](#jump5)
    - [模型推理](#jump6)
-   [模型推理性能](#jump7)



## <span id = "jump1">概述</span>

GhostNet基于Ghost模块，这个特点是不改变卷积的输出特征图的尺寸和通道大小，但是可以让整个计算量和参数数量大幅度降低。简单的说，GhostNet的主要贡献就是减低计算量、提高运行速度的同时，精准度降低的更少了，而且这种改变，适用于任意的卷积网络，因为它不改变输出特征图的尺寸
- 参考实现：
```
url=https://github.com/huawei-noah/CV-Backbones.git
branch=master
commit_id=5a06c87a8c659feb2d18d3d4179f344b9defaceb
model_name=GhostNet
```

##  输入输出数据

- 输入数据

| 输入数据 | 数据类型 | 大小 | 数据排布格式 |
| :-----| ----: |:---:|:------:|
| input | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW|

- 输出数据

| 输出数据  | 数据类型 | 大小 | 数据排布格式 |
|:------| ----: |:---:|:------:|
| output1 | FLOAT32 | 1 x 1000 | ND|

## <span id = "jump2">推理环境准备[所有版本]</span>

- 该模型需要以下插件与驱动  

    表 1 版本配套表

| 配套  |   版本    |                                                    环境准备指导                                                     |
|:------|:-------:|:-------------------------------------------------------------------------------------------------------------:|
| 固件与驱动 | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies/pies_00001.html) |
| CANN | 5.1.RC1 |                                                       -                                                       |
| Python |  3.7.5  |                                                       -                                                       |
| PyTorch |  1.8.0  |                                                       -                                                       |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 |    \    |                                                       \                                                       |

## <span id = "jump3">快速上手</span>
1.安装依赖
```
pip3 install -r requirment.txt
```
### <span id = "">获取源码</span>
1.下载开源仓
```
git clone https://github.com/huawei-noah/CV-Backbones.git
cd CV-Backbones
git reset --hard 5a06c87a8c659feb2d18d3d4179f344b9defaceb
cd ..
```
### <span id = "jump5">准备数据集</span>
1.获取原始数据集

本模型支持ImageNet 50000张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。

2.数据预处理

将原始数据（.jpeg）转化为二进制文件（.bin）。
执行imagenet_torch_preprocess.py脚本。
```
python3.7 imagenet_torch_preprocess.py resnet /home/HwHiAiUser/dataset/ImageNet/ILSVRC2012_img_val ./prep_dataset
```
第一个参数为模型类型，第二个参数为原始数据验证集（.jpeg）所在路径，第三个参数为输出的二进制文件（.bin）所在路径。
### <span id = "jump6">模型推理</span>
1.模型转换

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

a.获取权重文件

通过链接：http://github.com/huawei-noah/CV-Backbones/raw/master/ghostnet_pytorch/models/state_dict_73.98.pth
获取GhostNet1.0x权重文件state_dict_73.98.pth。

b.导出onnx文件
```
python3.7 ghostnet_pth2onnx.py state_dict_73.98.pth ghostnet.onnx
```
运行成功后生成ghostnet.onnx模型文件。

c.使用ATC工具将ONNX模型转OM模型

i.设置环境变量
```
source env.sh
```
ii.使用atc将onnx模型转换为om模型文件
```
atc --framework=5 --model=./ghostnet.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=ghostnet_bs1 --log=debug --soc_version=Ascend${chip_name}
```
参数说明： 

--model：为ONNX模型文件

--framework：5代表ONNX模型

--output：输出的OM模型

--input_format：输入数据的格式

--input_shape：输入数据的shape

--log：日志级别

--soc_version：处理器型号

运行成功后生成ghostnet_bs1.om模型文件

2.开始推理验证

a.使用ais_infer工具进行推理

i.安装包和对应安装方法参考链接：https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer

ii.设置环境变量
```
source env.sh
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/Ascend/driver/lib64/driver/
```

iii.将对应的ais_infer下载到相应目录下，执行ais_infer.py即可进行模型推理
```
python /home/infname46/ais_infer/ais_infer.py --model ./ghostnet_bs1.om --input ./prep_dataset/ --output ./ --outfmt NPY --batchsize 1
```
参数说明： 

--model：为OM模型文件

--input：为数据路径

--output：输出推理结果

--outfmt：输出结果的格式

--input_shape：输入数据的shape

--batchsize：模型接受的bs大小

### <span id = "jump7">模型推理性能</span>
1.精度验证

调用imagenet_acc_eval.py脚本与数据集标签val_label.txt比对，可以获得Accuracy数据，结果保存在result.json中
```
python3.7 imagenet_acc_eval.py ./lcmout/2022_xx_xx-xx_xx_xx/sumary.json /home/HwHiAiUser/dataset/imageNet/val_label.txt
```
参数说明：

第一项参数为推理结果中的sumary.json文件，第二项为gt标签文件

2.验证结果

GhostNet在310上的精度复现与性能表现如下表1，表2

表1-精度对比

| 芯片型号 | top1 | top5 |
|:------|:---:|:------:|
| 310 | 0.7398 | 0.9146 |
| 310P | 0.7398 | 0.9146 |

表2-性能对比

| Batch Size | 310 | 310P | t4 | 310P/310| 310P/t4|
|:------|:------:|:------:|:------:|:------:|:------:|
| 1 | 	1348.024 | 1502.4291 | 219.2172| 1.1145 | 6.8536|
| 4 | 2233.9991 | 2317.6152 | 701.0072 | 1.0374| 3.3061|
| 8 | 2463.9302 | 3739.9555| 1032.52| 1.5179 | 3.6222|
| 16 | 2624.8900 | 3438.7936| 924.992| 1.3101| 3.7176|
| 32 | 2689.0490 | 3020.9916| 447.872| 1.1234| 6.7452|

目前性能和精度都已达标，310P/310的最优batch为bs8，310P/t4最优为bs32