# BigGAN ONNX模型端到端推理指导
- [1. 模型概述](#1)
    - [论文地址](#11)
    - [代码地址](#12)
- [2. 环境说明](#2)
    - [深度学习框架](#21)
    - [python第三方库](#22)
- [3. 模型转换](#3)
    - [pth转onnx模型](#31)
    - [onnx转om模型](#32)
- [4. 输入数据生成](#4)
    - [数据生成](#41)
- [5. 离线推理](#5)
    - [msame工具概述](#51)
    - [离线推理](#52)
- [6. 精度对比](#6)
    - [模型后处理](#61)
    - [精度计算](#62)

## <a name="1">1. 模型概述</a>
### <a name="11">1.1 论文地址</a>
[BigGAN论文](https://arxiv.org/pdf/1809.11096.pdf)
### <a name="12">1.2 代码地址</a>
[BigGAN代码](https://github.com/ajbrock/BigGAN-PyTorch)

修改源码中的BigGAN.py、layers.py和inception_utils.py，并移至本项目中：
```
git clone https://github.com/ajbrock/BigGAN-PyTorch.git
mv biggan.patch BigGAN-PyTorch
cd BigGAN-PyTorch
git apply biggan.patch
scp BigGAN.py ..
scp layers.py ..
scp inception_utils.py ..
cd ..
```

## <a name="2">2. 环境说明</a>
### <a name="21">2.1 深度学习框架</a>

```
CANN 5.0.3
torch==1.8.0
torchvision==0.9.0
onnx==1.9.0
```
### <a name="22">2.2 python第三方库</a>

```
numpy
onnxruntime
scipy==1.7.1
onnx-simplifier==0.3.6
onnxoptimizer==0.2.6
```

 **说明：**
> PyTorch版本: 请不要低于1.6.0，否则在.pth文件转.onnx文件的过程中会产生报错  
> pytorch，torchvision和onnx:(X86架构)可以通过官方下载whl包安装; (Arm架构)可以通过源码编译安装   
> 其他第三方库: 可以通过 pip3.7 install -r requirements.txt 进行安装

## <a name="3">3. 模型转换</a>
一步式从pth权重文件转om模型的脚本，能够由pth权重文件生成bacth分别为1和16的om模型：
```bash
bash ./test/pth2om.sh
```
 **说明：**
> pth2om.sh中的6-14行: 完成pth转原始onnx模型  
> pth2om.sh中的18-29行: 完成onnx模型的简化，以及简化的onnx模型转om模型   

运行后会生成如下文件：
```bash
├── biggan.onnx
├── biggan_sim_bs1.onnx
├── biggan_sim_bs16.onnx
├── biggan_sim_bs1.om
├── biggan_sim_bs16.om
```

### <a name="31">3.1 pth转onnx模型</a>
1. 下载pth权重文件

[BigGAN预训练pth权重文件](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/gan/BigGAN/G_ema.pth)
>  **说明**
> 模型使用的权重文件名为G_ema.pth  

[Inception_v3预训练pth权重文件](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/gan/BigGAN/inception_v3_google.pth)
>  **说明**
> 下载的权重文件名为inception_v3_google.pth，此模型权重用于IS评价指标的计算，若仅进行图像生成，无需下载此权重文件 

[ImageNet采样的npz数据](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/gan/BigGAN/I128_inception_moments.npz)
>  **说明**
> 采样数据名为I128_inception_moments.npz，此数据用于FID评价指标的计算，若仅进行图像生成，无需下载此数据文件 

2. 执行biggan_pth2onnx.py脚本，生成onnx模型文件 
```bash
python3.7 biggan_pth2onnx.py --source "./G_ema.pth" --target "./biggan.onnx"
```
若需要修改pth2onnx部分，请注意目前ATC支持的onnx算子版本为11  

3. 执行clip_edit.py脚本，通过"input-model"和"output-model"参数指定输入和输出的onnx模型，默认输入输出均为"./biggan.onnx"
```bash
python3.7 clip_edit.py
```
>  **说明**
> 执行clip_edit.py目的在于初始化onnx模型中Clip节点中的"max"输入，便于后续onnx模型的简化

### <a name="32">3.2 onnx转om模型</a>
1. 使用onnx-simplifier简化onnx模型  
生成batch size为1的简化onnx模型，对应的命令为：
```bash
python3.7 -m onnxsim './biggan.onnx' './biggan_sim_bs1.onnx' --input-shape "noise:1,1,20" "label:1,5,148"
```

2. 设置环境变量

```bash
source env.sh
```

3. 使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)

```bash
atc --framework=5 --model=./biggan_sim_bs1.onnx --output=./biggan_sim_bs1 --input_format=ND --input_shape="noise:1,1,20;label:1,5,148" --log=error --soc_version=Ascend310
```

## <a name="4">4. 数据预处理</a>
- [输入数据生成](#41)
### <a name="41">4.1 数据生成</a>
1. BigGAN模型的输入数据是由噪声数据和标签数据组成，其中噪声数据是由均值为0，方差为1的正态分布中采样，标签数据是由0至类别总数中随机采样一个整数
2. 执行输入数据的生成脚本，生成模型输入的bin文件

```bash
#注：针对不同batch size的om模型需要生成不同的输入数据
python3.7 biggan_preprocess.py --batch-size 1 --num-inputs 50000
```
运行后，将会得到如下形式的文件夹：

```
├── prep_label_bs1
│    ├──input_00000.bin
│    ├──......
│
├── prep_noise_bs1
│    ├──input_00000.bin
│    ├──......         	 
```

## <a name="5">5. 离线推理</a>
执行一步式推理前，请先按照5.1节所示准备msame离线推理工具  
一步式进行输入数据的准备，模型离线推理和NPU性能数据的获取(针对batch1和batch16)：
```bash
bash ./test/eval_perf.sh
```
运行后会生成如下文件/文件夹：
```bash
├── prep_label_bs1    # 模型的标签输入(文件夹)
├── prep_label_bs16
├── prep_noise_bs1    # 模型的噪声输入(文件夹)
├── prep_noise_bs16
├── outputs_bs1_om    # 模型的输出(文件夹)
├── outputs_bs16_om
├── gen_y_bs1.npz     # 类别采样的npz数据
├── gen_y_bs16.npz
├── msame_bs1.txt     # msame推理过程的输出
├── msame_bs16.txt
├── bs1_perf.log      # 性能数据
├── bs16_perf.log
```
### <a name="51">5.1 msame工具概述</a>
msame模型推理工具，其输入是om模型以及模型所需要的输入bin文件，其输出是模型根据相应输入产生的输出文件。获取工具及使用方法可以参考[msame模型推理工具指南](https://gitee.com/ascend/tools/tree/master/msame)
### <a name="52">5.2 离线推理</a>
1. 设置环境变量
```bash
source env.sh
```
2. 执行离线推理
运行如下命令进行离线推理：
```bash
./msame --model "./biggan_sim_bs1.om" --input "./prep_noise_bs1,./prep_label_bs1"  --output "./outputs_bs1_om" --outfmt BIN > ./msame_bs1.txt
```
模型输出格式是bin，输出保存在"output"参数指定的文件夹中，同时会生成推理的日志文件msame_bs1.txt
3. 性能数据的获取
通过给test/parser.py指定推理后的日志文件，可以得到离线推理的性能数据
```bash
python3.7 ./test/parse.py --txt-file "./msame_bs1.txt" --batch-size 1 > bs1_perf.log
```
|模型|t4性能|310性能|
|----|----|----|
|BigGAN bs1|239.249fps|227.144fps|
|BigGAN bs16|344.900fps|282.898fps|

## <a name="6">6. 精度对比</a>
一步式进行输出数据的后处理和生成图像的评价指标(针对batch1和batch16)：
```bash
bash ./test/eval_acc.sh
```
运行后会生成如下文件/文件夹：
```bash
├── postprocess_img           # 转换后的模型输出(文件夹)
├── gen_img_bs1.npz           # 模型输出的npz数据
├── gen_img_bs16.npz
├── biggan_acc_eval_bs1.log   # 精度测量结果
├── biggan_acc_eval_bs16.log
```
### <a name="61">6.1 模型后处理</a>
模型后处理将离线推理得到的bin文件转换为jpg图像文件，并将原始输出保存至npz文件中，用于精度数据的获取
```
python3.7 biggan_postprocess.py --result-path "./outputs_bs1_om" --save-path "./postprocess_img" --batch-size 1 --save-img --save-npz
```
其中"result-path"表示离线推理输出所在的文件夹，"save-path"表示转换后图像文件的存储地址
### <a name="62">6.2 精度计算</a>
精度计算利用biggan_eval_acc.py脚本：
```bash
python3.7 biggan_eval_acc.py --num-inception-images 50000 --batch-size 1 --dataset 'I128' > biggan_acc_eval_bs1.log
```
其中"num-inception-images"表示用于进行精度测量的输出数量，"dataset"指定用于对比分布所采用的数据集，I128表示ImageNet数据集在train上的采样
>  **说明**  
> IS是生成图像的清晰度和多样性指标，其值越大说明越优  
> FID是生成图像集与真实图像集间的相似度指标，其值越小说明越优

| 模型 | IS | FID |
|-------|-------|-------|
|pth模型推理结果|94.323+/-2.395|9.9532|
|om模型离线推理结果|94.009+/-1.626|10.0411|