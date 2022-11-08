<<<<<<< HEAD
# EfficientDet D7 

### 推理流程
#### 环境准备
python3.7 install -r requirement.txt
```
克隆本代码仓后
cd EfficientDetD7
git clone https://github.com/rwightman/efficientdet-pytorch.git
cd efficientdet-pytorch
git checkout c5b694aa34900fdee6653210d856ca8320bf7d4e
patch -p1 < ../effdet.patch

cd ..
git clone https://gitee.com/zheng-wengang1/onnx_tools.git
cd onnx_tools
git checkout cbb099e5f2cef3d76c7630bffe0ee8250b03d921

d7.pth权重文件获取
链接: https://pan.baidu.com/s/1rDt4I9yobrApJqQFS13n3A  密码: 1f47
下载完成后
mkdir model
mv EfficientDet_file/d7.pth ./model
rm -r EfficientDet_file
coco数据集下载地址：http://images.cocodataset.org/zips/val2017.zip
标注json文件下载地址：http://images.cocodataset.org/annotations/annotations_trainval2017.zip
下载完成后
unzip val2017.zip
unzip annotations_trainval2017.zip
mkdir coco_data
mv annotations coco_data
mv val2017 coco_data
coco_data目录结构需满足:
coco_data
    ├── annotations
    └── val2017
```

#### 导出onnx模型
注：d7模型参数及输入shape过大，只能进行batch_size=1的推理
* 1、运行pth2onnx.py
```
batch_size=1
python3.7 pth2onnx.py --batch_size=1 --checkpoint=./model/d7.pth --out=./model/d7_bs1.onnx 
```
参数解释：  
batch_size:根据需求选择batch大小
checkpoint：pytorch权重文件  
out：输出onnx模型  
* 2、精简优化网络
```
pip3 install onnx-simplifier
batch_size=1
python3.7 -m onnxsim --input-shape="1,3,1536,1536" --dynamic-input-shape ./model/d7_bs1.onnx ./model/d7_bs1_sim.onnx
```
* 3、部分pad算子constant_value值-inf修改为0
```
batch_size=1
python3.7 modify_onnx.py --model=./model/d7_bs1_sim.onnx --out=./model/d7_bs1_modify.onnx
```
#### 转为om模型
* 使用Ascend atc工具将onnx转换为om
```
CANN安装目录
source /usr/local/Ascend/ascend-toolkit/set_env.sh
将atc日志打印到屏幕
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
设置日志级别
#export ASCEND_GLOBAL_LOG_LEVEL=0 #debug 0 --> info 1 --> warning 2 --> error 3
开启ge dump图
#export DUMP_GE_GRAPH=2
参考命令
batch_size=1
atc --framework=5 --model=./model/d7_bs1_modify.onnx --output=./model/d7_bs1 --input_format=NCHW --input_shape="x.1:1,3,1536,1536" --log=debug --soc_version=Ascend310
```
说明：  

1.--input_shape是模型输入节点的shape，可使用netron查看onnx输入节点名与shape，batch维值为16，即会生成batch size为16的om模型。无论onnx模型的batch是多少，只要通过--input_shape指定batch为正整数，就得到对应batch size的om模型，om模型虽然支持动态batch，但是我们不使用动态batch的om模型  
2.--out_nodes选项可以指定模型的输出节点，形如--out_nodes="节点1名称:0;节点2名称:0;节点3名称:0"就指定了这三个节点每个节点的第1个输出作为模型的第一，第二，第三个输出  
3.算子精度通过参数--precision_mode选择，默认值force_fp16  
3.开启autotune方法：添加--auto_tune_mode="RL,GA"  
5.开启repeat autotune方法：添加--auto_tune_mode="RL,GA"同时export REPEAT_TUNE=True  
6.配置环境变量ASCEND_SLOG_PRINT_TO_STDOUT和ASCEND_GLOBAL_LOG_LEVEL，然后执行命令atc ... > atc.log可以输出日志到文件  
7.配置环境变量DUMP_GE_GRAPH后执行atc命令时会dump中间过程生成的模型图，使用华为修改的netron可以可视化这些.pbtxt模型文件，如需要请联系华为方，当atc转换失败时可以查看ge生成的中间过程图的模型结构与算子属性，分析出哪个算子引起的问题  
8.如果使用aipp进行图片预处理需要添加--insert_op_conf=aipp_efficientnet-b0_pth.config  
9.atc工具的使用可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01  
10.若模型包含atc不支持的算子，算子问题可以规避的先通过修改模型进行规避，并在modelzoo上提issue或联系华为方  

#### 测试集预处理
coco val 5k 数据集下载
```
cd EfficientDetD7
mkdir bin_save
python3.7 preprocess.py --root=coco_data --bin-save=bin_save
python3.7 get_info.py bin ./bin_save ./d7_bin.info 1536 1536
```
选择coco val数据集进行验证  
参数解释：  
root:coco 验证集的路径   
bin_save：处理完bin文件存放路径  
预处理后的数据集信息文件d0_bin.info:
```
0 ./bin_save/000000184384.bin 1536 1536  
1 ./bin_save/000000182805.bin 1536 1536  
2 ./bin_save/000000223182.bin 1536 1536  
...
```
第一列为样本序号，第二列为预处理后的样本路径，第三四列为预处理后样本的宽高

#### 离线推理

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.1 推理benchmark工具用户指南 01
将benchmark工具放置于当前目录，与model同级
* 二进制输入
```
batch_size=1
./benchmark.x86_64 -model_type=vision -batch_size=1 -device_id=1 -om_path=./model/d7_bs1.om -input_text_path=d7_bin.info -input_width=1536 -input_height=1536 -output_binary=True -useDvpp=False
```

说明：
-model_type为benchmark支持的模型类型，目前支持的有vision，nmt，widedeep，nlp，yolocaffe，bert，deepfm  
-device_id是指运行在ascend 310的哪个device上，每张ascend 310卡有4个device  
-batch_size是指om模型的batch大小，该值应与om模型的batch大小相同，否则报输入大小不一致的错误  
-om_path是om模型文件路径  
-input_text_path为包含数据集每个样本的路径与其相关信息的数据集信息文件路径  
-input_height为输入高度  
-input_width为输入宽度  
-output_binary为以预处理后的数据集为输入，benchmark工具推理om模型的输出数据保存为二进制还是txt，但对于输出是int64类型的节点时，指定输出为txt时会将float类型的小数转换为0而出错  
-useDvpp为是否使用aipp进行数据集预处理  

#### 测试数据后处理
```
python3..7 postprocess.py --root=./coco_data --omfile=./result/dumpOutput_device1
```
参数解释：  
root：测试集目录   
omfile:om推理出的数据存放路径  

## EfficientDet-D7测试结果:

#### 性能测试
```
batch_size=1
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_d7_bs1_modify_in_device_1.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 1.50578samples/s, ave_latency: 664.142ms

GPU在线推理
cd efficientdet-pytorch 
python3.7 validate.py ./coco_data --model=tf_efficientdet_d7 --b=1 --checkpoint=d7.pth
在GPU上batch_size=1，推理速度
Test: [4270/5000]  Time: 0.964s (0.950s,    1.05/s)
Test: [4280/5000]  Time: 0.911s (0.950s,    1.05/s)
Test: [4290/5000]  Time: 0.977s (0.950s,    1.05/s)
Test: [4300/5000]  Time: 0.960s (0.950s,    1.05/s)
Test: [4310/5000]  Time: 0.939s (0.950s,    1.05/s)
Test: [4320/5000]  Time: 0.957s (0.950s,    1.05/s)
Test: [4330/5000]  Time: 0.951s (0.950s,    1.05/s)
Test: [4340/5000]  Time: 0.929s (0.950s,    1.05/s)
Test: [4350/5000]  Time: 0.956s (0.950s,    1.05/s)
Test: [4360/5000]  Time: 0.974s (0.950s,    1.05/s)
Test: [4370/5000]  Time: 0.955s (0.950s,    1.05/s)
Test: [4380/5000]  Time: 0.963s (0.950s,    1.05/s)
Test: [4390/5000]  Time: 0.949s (0.950s,    1.05/s)
Test: [4400/5000]  Time: 0.950s (0.950s,    1.05/s)
Test: [4410/5000]  Time: 0.924s (0.950s,    1.05/s)
Test: [4420/5000]  Time: 1.000s (0.950s,    1.05/s)
Test: [4430/5000]  Time: 0.977s (0.950s,    1.05/s)
Test: [4440/5000]  Time: 0.923s (0.950s,    1.05/s)
Test: [4450/5000]  Time: 0.918s (0.950s,    1.05/s)
Test: [4460/5000]  Time: 0.946s (0.950s,    1.05/s)
Test: [4470/5000]  Time: 0.958s (0.950s,    1.05/s)
Test: [4480/5000]  Time: 0.978s (0.950s,    1.05/s)
Test: [4490/5000]  Time: 0.976s (0.950s,    1.05/s)
对比：1.50*4>1.05
```
=======
# EfficientDetD7 ONNX模型端到端推理指导

- [1 模型概述](#1-模型概述)
    -   [1.1 论文地址](#11-论文地址)
    -   [1.2 代码地址](#12-代码地址)
- [2 环境说明](#2-环境说明)
    -   [2.1 深度学习框架](#21-深度学习框架)
    -   [2.2 python第三方库](#22-python第三方库)
- [3 模型转换](#3-模型转换)
    -   [3.1 pth转onnx模型](#31-pth转onnx模型)
    -   [3.2 onnx转om模型](#32-onnx转om模型)
- [4 数据集预处理](#4-数据集预处理)
    -   [4.1 数据集获取](#41-数据集获取)
    -   [4.2 数据集预处理](#42-数据集预处理)
- [5 离线推理](#5-离线推理)
    -   [5.1 构建ais_infer工具](#51-构建ais_infer工具)
    -   [5.2 离线推理](#52-离线推理)
- [6 精度对比](#6-精度对比)
    -   [6.1 离线推理精度](#61-离线推理精度)
    -   [6.2 精度对比](#62-精度对比)
- [7 性能对比](#7-性能对比)
    - [7.1 获取npu性能](#71-获取npu性能)
    - [7.2 获取T4性能](#72-获取T4性能)
    - [7.3 性能数据](#73-性能数据)
    - [7.4 性能对比](#74-性能对比)





## 1 模型概述

EfficientDet该论文首先提出了一种加权双向特征金字塔网络（BiFPN），它允许简单、快速的多尺度特征融合；其次，提出了一种复合特征金字塔网络缩放方法，统一缩放所有backbone的分辨率、深度和宽度、特征网络和box/class预测网络。

当融合不同分辨率的特征时，一种常见的方法是首先将它们调整到相同的分辨率，然后将它们进行总结。金字塔注意网络global self-attention上采样恢复像素定位。所有以前的方法都一视同仁地对待所有输入特征。 然而，论文中认为由于不同的输入特征在不同的分辨率，他们通常贡献的输出特征不平等。为了解决这个问题，论文建议为每个输入增加一个权重，并让网络学习每个输入特性的重要性。

### 1.1 论文地址

[Tan M, Pang R, Le Q V . EfficientDet: Scalable and Efficient Object Detection[C] 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020.](https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.html)

### 1.2 代码地址

```shell
ur=https://github.com/rwightman/efficientdet-pytorch
branch=master
commit_id=c5b694aa34900fdee6653210d856ca8320bf7d4e
```

## 2 环境准备 

### 2.1 深度学习框架

```
CANN 6.0.RC1
pytorch == 1.8.0
torchvision == 0.9.0
onnx == 1.8.0
```

### 2.2 python第三方库

```
timm == 0.6.1
pyyaml == 5.3.1
numpy == 1.20.0
pycocotools == 2.0.4
omegaconf == 2.2.2
onnx-simplifier == 0.4.5
skl2onnx == 1.10.4
backports.lzma == 0.0.14
```

**说明：** 

>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

> 由于设备显存限制，该模型只支持bs1。

### 3.1 pth转onnx模型

1. 准备pth权重文件  
   使用训练好的pkl权重文件：d7.pth

   下载地址： [链接: https://pan.baidu.com/s/1rDt4I9yobrApJqQFS13n3A  密码: 1f47](https://pan.baidu.com/s/1rDt4I9yobrApJqQFS13n3A)

   请在EfficientDetD7目录下创建model文件夹，并将d7.pth文件移入model文件夹中

2. 安装efficientdet-pytorch与onnx_tools

   ```
   git clone https://github.com/rwightman/efficientdet-pytorch.git
   cd efficientdet-pytorch
   git checkout c5b694aa34900fdee6653210d856ca8320bf7d4e
   patch -p1 < ../effdet.patch
   cd ..
   git clone https://gitee.com/zheng-wengang1/onnx_tools.git
   cd onnx_tools
   git checkout cbb099e5f2cef3d76c7630bffe0ee8250b03d921
   cd ../
   sed -i -e 's/onnx.onnx_ml_pb2/onnx/g' onnx_tools/OXInterface/OXInterface.py
   ```

3. 导出onnx文件

   1. 转出原始onnx文件

      ```
      python3.7 pth2onnx.py --batch_size=1 --checkpoint=./model/d7.pth --out=./model/d7_bs1.onnx 
      ```

      参数说明：

      - batch_size：模型转换的batch大小，目前仅支持1
      - checkpoint：pytorch权重文件  
      - out：输出onnx模型  

   2. 精简优化网络

      ```
      python3.7 -m onnxsim --input-shape="1,3,1536,1536" --dynamic-input-shape ./model/d7_bs1.onnx ./model/d7_bs1_sim.onnx --skip-shape-inference
      ```

   3. 将部分pad算子constant_value值-inf修改为0

      ```
   python3.7 modify_onnx.py --model=./model/d7_bs1_sim.onnx --out=./model/d7_bs1_modify.onnx
      ```
   
   最终，会在./model/目录下生成d7_bs1_modify.onnx作为最终得到的onnx文件

   

### 3.2 onnx模型转om模型

使用ATC工具将ONNX模型转OM模型。

1. 配置环境变量。

   ```shell
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

2. 使用atc将onnx模型
   ${chip_name}可通过npu-smi info指令查看，例：310P3
   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

执行ATC命令：

```shell
atc --framework=5 --model=./model/d7_bs1_modify.onnx --output=./model/d7_bs1 --input_format=NCHW --input_shape="x.1:1,3,1536,1536" --log=debug --soc_version=Ascend310P3
```

参数说明：

- model：为ONNX模型文件

- framework：5代表ONNX模型

- output：输出的OM模型

- input_format：输入数据的格式

- input_shape：输入数据的shape

- log：日志级别

- soc_version：处理器型号

## 4 数据集预处理

### 4.1 数据集获取

本模型支持coco2017 val 5000张图片的验证集。请用户自行获取数据集，上传数据集到代码仓目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到coco val2017.zip验证集及~/annotations中的instances_val2017.json数据标签。

数据目录结构请参考：

```
coco_data
├──val2017
├── annotations
    ├── instances_val2017.json
```

请将coco_data文件夹放在EfficientDetD7目录下。

### 4.2 数据预处理

数据预处理将原始数据集转换为模型输入的数据。

执行“preprocess.py”脚本，完成预处理。

```shell
python3.7 preprocess.py --root=coco_data --bin-save=bin_save
```

- root：coco数据集文件

- bin-save：输出的二进制文件（.bin）所在路径

每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成bin_save文件夹。

## 5 离线推理

### 5.1 构建[ais_infer工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)

1. 本推理工具编译需要安装好CANN环境。用户可以设置CANN_PATH环境变量指定安装的CANN版本路径，比如export CANN_PATH=/xxx/nnae/latest/. 如果不设置，本推理工具默认会从 CANN_PATH /usr/local/Ascend/nnae/latest/ /usr/local/Ascend/ascend-toolkit/latest 分别尝试去获取
2. 请克隆ais_infer代码仓库并进入ais-bench/tool/ais_infer目录，执行如下命令进行编译，即可生成推理后端whl包

```
git clone https://gitee.com/ascend/tools.git
cd  tools/ais-bench_workload/tool/ais_infer/backend
pip3.7 wheel ./
```

3. 在运行设备上执行如下命令，进行安装

```
pip3.7 install ./aclruntime-0.0.1-cp37-cp37m-linux_aarch64.whl
```

如果安装提示已经安装了相同版本的whl，请执行命令请添加参数"--force-reinstall"

```
pip3.7 install ./aclruntime-0.0.1-cp37-cp37m-linux_aarch64.whl --force-reinstall
```

4. 当whl包安装完成后，请在EfficientDetD7目录下进行如下操作:

```
git clone https://gitee.com/ascend/tools.git
cd tools
mv ais-bench_workload/tool/ais_infer/ais_infer.py ais-bench_workload/tool/ais_infer/frontend/ ../
cd ../
```



### 5.2 离线推理

1.设置环境变量

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.执行离线推理
执行推理前请使用npu-smi info查看设备状态，确保device空闲。

```shell
rm -rf ./result && mkdir result
python3.7 ais_infer.py --model model/d7_bs1.om --input ./bin_save --output ./result --outfmt BIN --batchsize=1 --infer_queue_count=1
```

参数说明：

- model：模型类型。

- input：经过预处理后的bin文件路径

- output：输出文件路径

- outfmt：输出文件格式

- batchsize：批次大小

- infer_queue_count: 推理队列的数据最大数量

推理后的结果会保存在./result目录下

## 6 精度对比

### 6.1 离线推理精度

调用“postprocess.py”脚本即可获得最终mAP精度：

```shell
python3.7 postprocess.py --root=./coco_data --omfile=./result
```

参数说明：

- root：coco数据集路径

- omfile：模型推理结果

执行完后得到310P上的精度。

```
mAP: 53.0
```

### 6.2 精度对比

 **评测结果：**官网pth精度[mAP: 53.1](https://github.com/rwightman/efficientdet-pytorch)，  310离线推理精度mAP: 53.0，精度下降在1%范围之内，故精度达标。

## 7 性能对比

### 7.1 获取npu性能

**性能测试：** 测试npu性能要确保device空闲，使用npu-smi info命令可查看device是否在运行其它推理任务。性能测试可使用`ais_infer`工具。

```
python3.7 ais_infer.py --model model/d7_bs1.om --output ./ --outfmt BIN --loop 20 --batchsize=1
```

可以直接从**throughtput**中读取模型对应吞吐率。

`ais_infer`工具在整个数据集上推理方式测性能可能时间较长，纯推理方式测性能可能不准确，因此模型要使用在整个数据集上推理的方式测性能。

```
python3.7 ais_infer.py --model model/d7_bs1.om --input ./bin_save --output ./result --outfmt BIN --batchsize=1 --infer_queue_count=1
```

### 7.2 获取T4性能

在装有T4卡的服务器上使用`onnxruntime-gpu`工具测试gpu性能，测试代码如下。测试过程请确保卡没有运行其他任务。

```python
import time
from turtle import width
import numpy as np
import onnxruntime as rt
print(rt.get_device())

batch_size=1
length=1536
width=1536
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

sess = rt.InferenceSession("./model/d7_bs1_modify.onnx", providers=providers)
print("providers:",sess.get_providers())
input_name = sess.get_inputs()[0].name
outputs = ["output"]

# onnx模型输入节点
data = np.random.randn(batch_size,3, length, width).astype(np.float32)

# 推理200次，
for K in range(200):
    start_time = time.time()
    result = sess.run([], {input_name: data})
    end_time = time.time() - start_time
    time_list.append(end_time)

print("Batch_size: ",batch_size)
print("Time used: ", np.mean(time_list), 's')
print("T4 Throughput: ",batch_size/np.mean(time_list))
```

### 7.3 性能数据

| Throughput | 310  | 310P | T4    | 310P/310    | 310P/T4     |
| ---------- | ---- | ---- | ----- | ----------- | ----------- |
| bs1        | 7.32 | 6.18 | 1.689 | 0.844262295 | 3.658969805 |
| 最优batch  | 7.32 | 6.18 | 1.689 | 0.844262295 | 3.658969805 |

### 7.4 性能对比

性能达到T4性能的1.6倍，但在310P上的性能无法达到310的1.2倍，故性能不达标。
>>>>>>> E7 init
