# convmixer ONNX模型端到端推理指导
- [1. 模型概述](#1)
    - [论文地址](#11)
    - [代码地址](#12)
- [2. 环境说明](#2)
    - [深度学习框架](#21)
    - [python第三方库](#22)
- [3. 模型转换](#3)
    - [pth转onnx模型](#31)
    - [onnx转om模型](#32)
- [4. 数据预处理](#4)
    - [数据处理](#41)
- [5. 离线推理](#5)
    - [msame工具概述](#51)
    - [离线推理](#52)
    - [精度和性能比较](#53)

## <a name="1">1. 模型概述</a>
### <a name="11">1.1 论文地址</a>
[convmixer论文](https://openreview.net/forum?id=TVHS5Y4dNvM)
### <a name="12">1.2 代码地址</a>
[convmixer代码](https://github.com/locuslab/convmixer)

修改源码中的convmixer.py并移至本项目中重命名为convmixer_net.py：
```
git clone https://github.com/locuslab/convmixer.git
mv convmixer_patch.patch convmixer
cd convmixer
git apply convmixer_patch.patch
scp convmixer.py ../convmixer_net.py
cd ..
```
> **说明：**   
> 本离线推理项目中convmixer模型对应论文中convmixer_1536_20，以下说明中将convmixer_1536_20简称为convmixer

## <a name="2">2. 环境说明</a>
### <a name="21">2.1 深度学习框架</a>
```
CANN 5.0.2
torch==1.5.0
torchvision==0.6.0
onnx==1.10.2
```

### <a name="22">2.2 python第三方库</a>
```
numpy
Pillow==9.0.1
```
> **说明：**  
> pytorch，torchvision和onnx:(X86架构)可以通过pip方式安装或官方下载whl包安装; (Arm架构)可以通过源码编译安装   
> 其他第三方库: 可以通过 pip3.7 install -r requirements.txt 进行安装

## <a name="3">3. 模型转换</a>
一步式从pth.tar权重文件转om模型的脚本，能够由pth.tar权重文件生成bacth为1的om模型：
```bash
bash ./test/pth2om.sh --batch_size=1 --not_skip_onnx=true
```
运行后会生成如下文件：
```bash
├── convmixer_1536_20.onnx
├── convmixer_1536_20_bs1.om
```

### <a name="31">3.1 pth转onnx模型</a>
1. 设置环境变量
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 下载convmixer_1536_20的权重文件: 
[convmixer_1536_20_ks9_p7.pth.tar](https://github.com/tmp-iclr/convmixer/releases/download/v1.0/convmixer_1536_20_ks9_p7.pth.tar)

3. 执行convmixer_pth2onnx.py脚本，生成onnx模型文件 
```bash
python3.7 convmixer_pth2onnx.py --source "./convmixer_1536_20_ks9_p7.pth.tar" --target "./convmixer_1536_20.onnx"
```
其中"source"表示模型加载权重的地址和名称，"target"表示转换后生成的onnx模型的存储地址和名称  

### <a name="32">3.2 onnx转om模型</a>
1. 使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/51RC2alpha002/infacldevg/atctool)

```bash
atc --framework=5 --model=./convmixer_1536_20.onnx --output=./convmixer_1536_20_bs1 --input_format=NCHW --input_shape="image:1,3,224,224" --log=error --soc_version=Ascend710
```

## <a name="4">4. 数据预处理</a>
数据预处理过程包含在 test/eval_acc_perf.sh 的脚本中
### <a name="41">4.1 数据处理</a>
1. 设置环境变量
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. convmixer模型使用ImageNet2012中的5万张验证集数据进行测试，具体来说参考convmixer的源码仓中的测试过程对验证集图像进行缩放，中心裁剪以及归一化，并将图像数据转换为二进制文件(.bin)
> **说明：**  
> 本项目使用的推理工具为msame，需要针对不同的batch size生成不同的输入数据  

3. 执行输入数据的生成脚本，生成模型输入的bin文件
```bash
python3.7 convmixer_preprocess.py --image-path /opt/npu/imageNet/val --prep-image ./prep_image_bs1 --batch-size 1
```
其中"image-path"表示处理前原数据集的地址，"prep-image"表示生成数据集的文件夹名称(将在文件夹名称后会自动标识对应batch size)，"batch-size"表示生成数据集对应的batch size
运行后，将会得到如下形式的文件夹：
```
├── prep_image_bs1
│    ├──input_00000.bin
│    ├──......     	 
```

## <a name="5">5. 离线推理</a>
执行一步式推理前，请先按照5.1节准备msame离线推理工具  
一步式进行输入数据的准备，模型离线推理和NPU性能数据的获取：
```bash
bash ./test/eval_acc_perf.sh --batch_size=1 --datasets_path=/opt/npu/imageNet
```
运行后会生成如下文件/文件夹：
```bash
├── prep_image_bs1        # 模型的标签输入(文件夹)
├── msame_bs1.txt         # msame推理过程的输出
├── result            
│    ├── outputs_bs1_om   # 模型的输出(文件夹)
│    ├── result_bs1.json  # 模型的精度输出
```

### <a name="51">5.1 msame工具概述</a>
msame模型推理工具，其输入是om模型以及模型所需要的输入bin文件，其输出是模型根据相应输入产生的输出文件。获取工具及使用方法可以参考[msame模型推理工具指南](https://gitee.com/ascend/tools/tree/master/msame)
### <a name="52">5.2 离线推理</a>
1. 设置环境变量
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 执行离线推理
运行如下命令进行离线推理：
```bash
./msame --model ./convmixer_1536_20_bs1.om  --output ./result/outputs_bs1_om --outfmt TXT --input ./prep_image_bs1 > msame_bs1.txt
```
模型输出格式是txt，输出保存在"output"参数指定的文件夹中，同时会生成推理的日志文件msame_bs1.txt

### <a name="53">5.3 精度和性能比较</a>
1. 性能数据的获取
通过给test/parser.py指定推理后的日志文件，可以得到离线推理的性能数据
```bash
python3.7 test/parse.py --result-file ./msame_bs1.txt --batch-size 1
```
其中"result-file"表示性能数据的地址和名称，"batch-size"表示性能测试时模型对应的batch size

2. 精度数据的计算
精度计算利用convmixer_eval_acc.py脚本
```
python3.7 convmixer_eval_acc.py --folder-davinci-target ./result/outputs_bs1_om/ --annotation-file-path /opt/npu/imageNet/val_label.txt --result-json-path ./result --json-file-name result_bs1.json --batch-size 1
```
其中"folder-davinci-target"表示离线推理输出所在的文件夹，"annotation-file-path"表示ImageNet2012验证集标签的地址和名称，"result-json-path"输出精度数据所在的文件夹，"json-file-name"表示输出精度数据所在的文件名，"batch-size"表示精度测试时模型对应的batch size

3. 精度数据的获取
通过给test/parser.py指定推理后的日志文件，可以得到离线推理的性能数据
```bash
python3.7 test/parse.py --result-file ./result/result_bs1.json
```
| 模型      | 参考精度  | 710精度  | 性能基准    | 710性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| convmixer_1536_20 bs1  | top1:81.37%  | top1:81.35%  | 44.445fps | 67.114fps | 
| convmixer_1536_20 bs4 | top1:81.37%  | top1:81.35%  |59.353fps | 72.727fps | 

> **说明：**  
> Top1表示预测结果中概率最大的类别与真实类别一致的概率，其值越大说明分类模型的效果越优 