# PoseC3D ONNX模型端到端推理指导
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
[Revisiting Skeleton-based Action Recognition](https://arxiv.org/abs/2104.13586)

### <a name="12">1.2 代码地址</a>
[posec3d代码](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/posec3d)

```bash
git clone https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/posec3d.git
```
> **说明：**   
> 本离线推理项目中posec3d模型对应论文中PoseConv3D，以下说明中将PoseConv3D简称为posec3d

## <a name="2">2. 环境说明</a>
### <a name="21">2.1 深度学习框架</a>
```
CANN 5.1.RC1
torch==1.11.0
torchvision==0.12.0
onnx==1.11.0
```

### <a name="22">2.2 python第三方库</a>
```
numpy==1.21.6
pillow==9.1.1
mmcv==1.4.0
```
> **说明：**  
> pytorch，torchvision和onnx:(X86架构)可以通过pip方式安装或官方下载whl包安装; (Arm架构)可以通过源码编译安装   
> 其他第三方库: 可以通过 pip3.7 install -r requirements.txt 进行安装

## <a name="3">3. 模型转换</a>
一步式从pth权重文件转om模型的脚本，能够由pth权重文件生成bacth为1的om模型：
```bash
bash ./test/pth2om.sh --batch_size=1 --not_skip_onnx=true
```
运行后会生成如下文件：
```bash
├── posec3d_bs1.onnx
├── posec3d_bs1.om
```

### <a name="31">3.1 pth转onnx模型</a>
1. 设置环境变量
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 下载由hmdb51数据集训练得到的模型权重文件: 
[HMDB51-ckpt](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/posec3d)表格HMDB51对应ckpt(checkpoint)一栏的链接，点击即可直接下载。

3. 执行posec3d_pth2onnx.py脚本，生成onnx模型文件 
```bash
python ./posec3d_pytorch2onnx.py ./mmaction2/configs/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint.py ./slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint-76ffdd8b.pth --shape 1 20 17 48 56 56 --verify --output-file ./posec3d_bs1.onnx
```
其中"shape"表示输入节点的shape，"output-file"表示转换后生成的onnx模型的存储地址和名称  

### <a name="32">3.2 onnx转om模型</a>
1. 使用atc将onnx模型转换为om模型文件，posec3d模型需要借助aoe优化获得的知识库，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/51RC2alpha002/infacldevg/atctool)

```bash
# 将知识库文件夹通过scp -r指令复制到当前目录
export TUNE_BANK_PATH="./aoe_result_bs1"
atc --framework=5 --model=./posec3d_bs1.onnx --output=./posec3d_bs1 --input_format=ND --input_shape="invals:1,20,17,48,56,56" --log=debug --soc_version=Ascend710
```

## <a name="4">4. 数据预处理</a>
在当前目录下载hmdb51.pkl标注文件
[hmdb51](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/skeleton)在Prepare Annotations目录下附有直接下载hmdb51.pkl标注文件的链接

数据预处理过程包含在 test/eval_acc_perf.sh 的脚本中
### <a name="41">4.1 数据处理</a>
1. 设置环境变量
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 下载hmdb51数据集。posec3d模型使用hmdb51中的1530个视频数据进行测试，具体来说参考posec3d的源码仓中的测试过程对验证集视频进行提取帧，对预处理后的图像进行缩放，中心裁剪以及归一化，并将图像数据转换为二进制文件(.bin)
```bash
cd ./mmaction2/tools/data/hmdb51
bash download_annotations.sh
bash download_videos.sh
bash extract_rgb_frames_opencv.sh
bash generate_rawframes_filelist.sh
bash generate_videos_filelist.sh
mv ../../../data/hmdb51 /opt/npu
cd ../../../..
```
> **说明：**  
> 本项目使用的推理工具为msame，需要针对不同的batch_size生成不同的输入数据  

3. 执行输入数据的生成脚本，生成模型输入的bin文件
```bash
python3.7 posec3d_preprocess.py --batch_size 1 --data_root /opt/npu/hmdb51/rawframes/  --ann_file hmdb51.pkl --name /opt/npu/hmdb51/prep_hmdb51_bs1
```
其中"batch_size"表示生成数据集对应的batch size,data_root表示处理前原数据集的地址，ann_file表示对应的标注文件，"name"表示生成数据集的文件夹名称。
运行后，将会得到如下形式的文件夹：
```
├── prep_image_bs1
│    ├──0.bin
│    ├──......     	 
```

## <a name="5">5. 离线推理</a>
执行一步式推理前，请先按照5.1节准备msame离线推理工具  
一步式进行输入数据的准备，模型离线推理和NPU性能数据的获取：
```bash
bash ./test/eval_acc_perf.sh --batch_size=1 --datasets_path=/opt/npu/hmdb51
```
运行后会生成如下文件/文件夹：
```bash
├── hmdb51         		  # /opt/npu下数据集(文件夹)            
│    ├── annotations   	  # 注释文件(文件夹)
│    ├── prep_hmdb51_bs1  # 预处理输出的二进制文件(文件夹)       
│    ├── rawframes   	  # 原始帧(文件夹)       
│    ├── videos   		  # 视频数据(文件夹)
```
```bash
├── msame_bs1.txt         # msame推理过程的输出
├── result            
│    ├── outputs_bs1_om   # 模型的输出(文件夹)
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
./msame --model "./posec3d_bs1.om" --input "/opt/npu/hmdb51/prep_hmdb51_bs1" --output "./result/outputs_bs1_om" --outfmt BIN > msame_bs1.txt
```
模型输出格式是txt，输出保存在"output"参数指定的文件夹中，同时会生成推理的日志文件msame_bs1.txt

### <a name="53">5.3 精度和性能比较</a>
1. 性能数据的获取
通过给test/parser.py指定推理后的日志文件，可以得到离线推理的性能数据
```bash
python3.7 test/parse.py --result-file ./msame_bs1.txt --batch-size 1
```
其中"result-file"表示性能数据的地址和名称，"batch_size"表示性能测试时模型对应的batch size

2. 精度数据的计算
精度计算利用posec3d_postprocess.py脚本
```
 python3.7 posec3d_postprocess.py --result_path ./result/outputs_bs1_om/{实际文件夹名}
```
其中result_path表示离线推理输出所在的文件夹，info_path（默认为"./hmdb51.info"）表示hmdb51验证集标签的地址和名称。

| 模型      | 参考精度  | 310P精度  | 性能基准    | 310P性能    |
| :------: | :------: | :------: | :------:  | :------:  |
| posec3d_hmdb51_bs1  | top1:69.3%  | top1:69.2%  | 15.385fps | 24.699fps |

> **说明：**  
> Top1表示预测结果中概率最大的类别与真实类别一致的概率，其值越大说明分类模型的效果越优 