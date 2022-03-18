# SK-ResNet50 Onnx 模型端到端推理指导

- [1. 模型概述](#1)
  - [论文地址](#11)
  - [代码地址](#12)
- [2. 环境说明](#2)
  - [深度学习框架](#21)
  - [python第三方库](#22)
- [3. 模型转换](#3)
  - [pth转onnx模型](#31)
- [4. 数据预处理](#4)
  - [数据集获取](#41)
  - [数据集预处理](#42)
  - [生成数据集信息文件](#43)
- [5. 离线推理](#5)
  - [benchmark工具概述](#51)
  - [离线推理](#52)
- [6. 精度对比](#6)
  - [离线推理TopN精度](#61)
  - [精度对比](#62)
- [7. 性能对比](#7)
  - [npu性能数据](#71)

## <a name="1">1. 模型概述</a>

### <a name="11">1.1 论文地址</a>

[SK-ResNet 论文](https://arxiv.org/pdf/1903.06586.pdf) 

### <a name="12">1.2 代码地址</a>

[SK-ResNet 代码](https://github.com/implus/PytorchInsight)

branch: master

commit_id: 2864528f8b83f52c3df76f7c3804aa468b91e5cf

## <a name="2">2. 环境说明</a>

### <a name="21">2.1 深度学习框架</a>

```
pytorch == 1.5.0
torchvision == 0.6.0
onnx == 1.9.0
```

### <a name="22">2.2 python第三方库</a>

```
numpy == 1.19.2
Pillow == 8.2.0
opencv-python == 4.5.2
```

> **说明：**
>
> X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装 
>
> Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## <a name="3">3. 模型转换</a>

### <a name="31">3.1 pth转onnx模型</a>

1. 下载 pth 权重文件

   [SK-ResNet50预训练pth权重文件(百度网盘，提取码：tfwn)](https://pan.baidu.com/s/1Lx5CNUeRQXOSWjzTlcO2HQ)

   文件名：sk_resnet50.pth.tar

   md5sum：979bbb525ee0898003777a8e663e91c0

2. 克隆代码仓库代码

   ```bash
   git clone https://github.com/implus/PytorchInsight.git
   ```

3. 使用 sknet2onnx.py 转换pth为onnx文件，在命令行运行如下指令：

   ```bash
   python3.7 sknet2onnx.py --pth sk_resnet50.pth.tar --onnx sknet50_bs1
   ```
   
   sk_resnet50.pth.tar文件为步骤1中下载的预训练权重文件，该条指令将在运行处生成一个sknet50_bs1文件，此文件即为目标onnx文件

**模型转换要点：**

> pytorch导出onnx时softmax引入了transpose以操作任意轴，然而在onnx中已支持softmax操作任意轴，故可删除transpose提升性能

### <a name="32">3.2 onnx转om模型</a>

下列需要在具备华为Ascend系列芯片的机器上执行：

1. 设置 atc 工作所需要的环境变量

   ```bash
   export install_path=/usr/local/Ascend/ascend-toolkit/latest
   export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
   export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
   export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
   export ASCEND_OPP_PATH=${install_path}/opp
   ```

2. 使用atc工具将onnx模型转换为om模型，命令参考

   ```bash
   atc --framework=5 --model=sknet50.onnx --output=sknet50_bs1 --input_format=NCHW --input_shape="image:1,3,224,224" --log=debug --soc_version=Ascend310
   ```

   此命令将在运行路径下生成一个sknet50_1bs.om文件，此文件即为目标om模型文件

## <a name="4">4. 数据预处理</a>

### <a name="41">4.1 数据集获取</a>

该模型使用[ImageNet官网](http://www.image-net.org/)的5万张验证集进行测试，图片与标签分别存放在/opt/npu/imagenet/val与/opt/npu/imagenet/val_label.txt。

### <a name="42">4.2 数据集预处理</a>

使用 sknet_preprocess.py 脚本进行数据预处理，脚本执行命令：

```bash
python3.7 sknet_preprocess.py -s /opt/npu/imagenet/val -d ./prep_data
```

### <a name="43">4.3 生成数据集信息文件</a>

1. 生成数据集信息文件脚本 get_info.py

2. 执行生成数据集信息脚本，生成数据集信息文件

   ```bash
   python3.7 get_info.py bin ./prep_data ./sknet_prep_bin.info 224 224
   ```

   第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息

## <a name="5">5. 离线推理</a>

### <a name="51">5.1 benchmark工具概述</a>

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN V100R020C10 推理benchmark工具用户指南 01

### <a name="52">5.2 离线推理</a>

```bash
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=sknet50_bs1.om -input_text_path=sknet_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```

输出结果默认保存在当前目录result/dumpOutput_device{0}，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

## <a name="6">6. 精度对比</a>

### <a name="61">6.1 离线推理TopN精度</a>

后处理统计TopN精度，调用imagenet_acc_eval.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中：

```bash
python3.7 vision_metric_ImageNet.py result/dumpOutput_device0/ ../data/sknet/val_label.txt ./ result.json
```

第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。查看输出结果：

```json
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "77.54%"}, {"key": "Top2 accuracy", "value": "87.12%"}, {"key": "Top3 accuracy", "value": "90.73%"}, {"key": "Top4 accuracy", "value": "92.55%"}, {"key": "Top5 accuracy", "value": "93.71%"}]}
```

经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上

### <a name="62">6.2 精度对比</a>

|                    |   TOP1   |   TOP5   |
| :----------------: | :------: | :------: |
|  原github仓库精度  | 77.5380% | 93.7000% |
| om模型离线推理精度 |  77.54%  |  93.71%  |

将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。

## <a name="7">7. 性能对比</a>

### <a name="71">7.1 npu性能数据</a>

benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。

benchmark工具作纯推理时使用的命令参考如下：

```bash
./benchmark.x86_64 -round=20 -om_path=sknet50_bs1.om -batch_size=1 
```

1. batch1 性能

   使用benchmark工具在整个数据集上推理时获得的性能数据：

   ```
   [e2e] throughputRate: 143.402, latency: 348669
   [data read] throughputRate: 152.003, moduleLatency: 6.57881
   [preprocess] throughputRate: 151.416, moduleLatency: 6.60433
   [infer] throughputRate: 143.733, Interface throughputRate: 210.306, moduleLatency: 6.16176
   [post] throughputRate: 143.732, moduleLatency: 6.95737
   ```

   Interface throughputRate: 210.306 * 4 = 841.224 即是batch1 310单卡吞吐率

2. batch4 性能

   ```
   [INFO] ave_throughputRate: 315.424samples/s, ave_latency: 3.30141ms
   ```

   Interface throughputRate: 315.424 * 4 = 1261.696 即是batch4 310单卡吞吐率

3. batch8 性能

   ```
   [INFO] ave_throughputRate: 365.813samples/s, ave_latency: 2.76526ms
   ```

   Interface throughputRate: 365.813 * 4 = 1463.252 即是batch8 310单卡吞吐率

4. batch16 性能

   ```
   [e2e] throughputRate: 196.399, latency: 254584
   [data read] throughputRate: 208.891, moduleLatency: 4.78718
   [preprocess] throughputRate: 207.779, moduleLatency: 4.81281
   [infer] throughputRate: 197.514, Interface throughputRate: 392.072, modul
   [post] throughputRate: 12.3443, moduleLatency: 81.0088
   ```

   Interface throughputRate: 392.072 * 4 = 1568.288 即是batch16 310单卡吞吐率

5. batch32 性能

   ```
   [INFO] ave_throughputRate: 376.691samples/s, ave_latency: 2.66319ms
   ```

   Interface throughputRate: 376.691 * 4 =  1506.764 即是batch32 310单卡吞吐率

**性能优化**

> 从profiling数据的op_statistic_0_1.csv看出影响性能的是transpose算子，从onnx结构图看出该算子用于实现softmax任意轴，由pytorch导出时引入，然而softmax在onnx中现已支持任意轴，故可直接删除该算子提升性能，删除代码参考如下：

```python
model = onnx.load(args.onnx+'.onnx')
graph = model.graph
node = graph.node
softmax_node_index = []
del_group = []
for i in range(len(node)):
	if node[i].op_type == 'Softmax':
		del_group.append((node[i-1], node[i], node[i+1], i))
for g in del_group:
   new_input = g[0].input
   new_output = g[2].output
   new_name = g[1].name
   new_index = g[3]
   new_node = onnx.helper.make_node("Softmax", new_input, new_output, new_name, axis=1)
   for n in g[:-1]:
      graph.node.remove(n)
   graph.node.insert(new_index, new_node)
onnx.save(model, args.onnx+'.onnx')
```



