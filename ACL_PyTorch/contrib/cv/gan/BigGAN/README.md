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
    - [ais_infer工具概述](#51)
    - [离线推理](#52)
- [6. 精度对比](#6)
    - [模型后处理](#61)
    - [精度计算](#62)

## <a name="1">1. 模型概述</a>
### <a name="11">1.1 论文地址</a>
[BigGAN论文](https://arxiv.org/pdf/1809.11096.pdf)
### <a name="11">1.2 参考实现</a>
```
url=https://github.com/ajbrock/BigGAN-PyTorch	
branch=master 
commit_id=98459431a5d618d644d54cd1e9fceb1e5045648d
```
### <a name="12">1.2 代码地址</a>
[BigGAN代码](https://github.com/ajbrock/BigGAN-PyTorch)

修改源码中的BigGAN.py、layers.py和inception_utils.py，并移至本项目中：
```
git clone https://github.com/ajbrock/BigGAN-PyTorch.git
mv biggan.patch BigGAN-PyTorch
cd BigGAN-PyTorch
dos2unix biggan.patch
git apply biggan.patch
cp BigGAN.py ..
cp layers.py ..
cp inception_utils.py ..
cd ..
```

## <a name="2">2. 环境说明</a>
### <a name="21">2.1 深度学习框架</a>

```
CANN 5.1.RC1
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
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

3. 使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)

${chip_name}可通过npu-smi info指令查看，例：310P3
![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```bash
atc --framework=5 --model=./biggan_sim_bs1.onnx --output=./biggan_sim_bs1 --input_format=ND --input_shape="noise:1,1,20;label:1,5,148" --log=error --soc_version=Ascend${chip_name}
```

参数说明：

    --output：输出的OM模型。  
    --log：日志级别。 
    --soc_version：处理器型号，通过npu-smi info查询。  
    --input_format：输入数据格式。
    --input_shape：模型输入数据的shape。
   
运行后会生成如下文件：
```bash
├── biggan.onnx
├── biggan_sim_bs1.onnx
├── biggan_sim_bs1.om
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

### <a name="51">5.1 ais_infer工具概述</a>
a.推理模型使用ais_infer工具
```bash
git clone https://gitee.com/ascend/tools.git
```
   i.设置环境变量
   ```bash
   export DDK_PATH=/home/HwHiAiUser/Ascend/ascend-toolkit/latest
   export NPU_HOST_LIB=/home/HwHiAiUser/Ascend/ascend-toolkit/latest/acllib/lib64/stub
   ```
   以上为设置环境变量的示例，请将/home/HwHiAiUser/Ascend/ascend-toolkit/latest替换为Ascend 的ACLlib安装包的实际安装路径。
### <a name="52">5.2 离线推理</a>

1. 安装ais_infer推理工具
   安装链接: https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer
   
2.执行离线推理
·设置环境变量
```
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:${LD_LIBRARY_PATH}
```
·执行命令
```
    cd tools/ais-bench_workload/tool/ais_infer
    mkdir -p /home/ylz/BigGAN/outputs_bs1_om
    python3.7 ais_infer.py --model "./biggan_sim_bs1.om" --input "./prep_noise_bs1,./prep_label_bs1"  --output "./outputs_bs1_om" --outfmt BIN --batchsize 1 >./result1.txt
```

    --model ：输入的om文件。
    --input：输入的bin数据文件。
    --output：推理数据输出路径。
    --outfmt：输出数据的格式。
    --batchsize：模型batch size。
    
模型输出格式是bin，输出保存在"output"参数指定的文件夹中，同时会生成文件result1.txt

运行后会生成如下文件/文件夹：
```bash
├── prep_label_bs1    # 模型的标签输入(文件夹)
├── prep_noise_bs1    # 模型的噪声输入(文件夹)
├── outputs_bs1_om    # 模型的输出(文件夹)
├── gen_y_bs1.npz     # 类别采样的npz数据
├── result1.txt     # ais_infer推理过程的输出
├── bs1_perf.log      # 性能数据
```

|模型|t4性能|310性能|310P性能|
|----|----|----|----|
|BigGAN bs1|236fps|242fps|443fps|
|BigGAN bs16|334fps|333fps|544fps|

## <a name="6">6. 精度对比</a>

### <a name="61">6.1 模型后处理</a>
模型后处理将离线推理得到的bin文件转换为jpg图像文件，并将原始输出保存至npz文件中，用于精度数据的获取
```
cd /home/ylz/BigGAN
python3.7 biggan_postprocess.py --result-path "./outputs_bs1_om/2022_08_29-16_54_31" --save-path "./postprocess_img" --batch-size 1 --save-img --save-npz
```
其中"result-path"表示离线推理输出所在的文件夹，"save-path"表示转换后图像文件的存储地址
### <a name="62">6.2 精度计算</a>
精度计算利用biggan_eval_acc.py脚本：
```bash
python3.7 biggan_eval_acc.py --num-inception-images 50000 --batch-size 1 --dataset 'I128' > biggan_acc_eval_bs1.log
```
运行后会生成如下文件/文件夹：
```bash
├── postprocess_img           # 转换后的模型输出(文件夹)
├── gen_img_bs1.npz           # 模型输出的npz数据
├── biggan_acc_eval_bs1.log   # 精度测量结果
```

其中"num-inception-images"表示用于进行精度测量的输出数量，"dataset"指定用于对比分布所采用的数据集，I128表示ImageNet数据集在train上的采样
>  **说明**  
> IS是生成图像的清晰度和多样性指标，其值越大说明越优  
> FID是生成图像集与真实图像集间的相似度指标，其值越小说明越优

| 模型 | IS | FID |
|-------|-------|-------|
|pth模型推理结果|94.323+/-2.395|9.9532|
|om模型离线推理结果|94.009+/-1.626|10.0411|