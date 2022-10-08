## 1 模型概述
### 1.1 论文地址
[SE_ResNeXt50](https://arxiv.org/abs/1611.05431)  
SE_ResNeXt50_32x4d是一种用于图像分类的卷积神经网络，这个模型的默认输入尺寸是224×224，有三个通道。通过利用多路分支的特征提取方法，提出了一种新的基于ResNet残差模块的网络组成模块，并且引入了一个新的维度cardinality。该网络模型可以在于对应的ResNet相同的复杂度下，提升模型的精度（相对于最新的ResNet和Inception-ResNet)）同时，还通过实验证明，可以在不增加复杂度的同时，通过增加维度cardinality来提升模型精度，比更深或者更宽的ResNet网络更加高效。
### 1.2 代码地址
[SE_ResNeXt50](git clone https://github.com/Cadene/pretrained-models.pytorch.git)  
branch:master   
commit_id:8aae3d8f1135b6b13fed79c1d431e3449fdbf6e0 
  
## 2 环境说明
使用CANN版本为CANN:5.1.RC2

深度学习框架与第三方库
```
onnx=1.7.0
pytorch=1.6.0
torchVision=0.7.0
numpy=1.18.5
opencv-python=4.5.5.64
protobuf=3.19.0
```
**说明：** 
请用户根据自己的运行环境自行安装所需依赖。
## 3. 获取源码
```text
├── get_info.py                      //生成推理输入的数据集二进制info文件或jpg info文件
├── preprocess_se_resnext50_32x4d_pth.py   //数据集预处理脚本，通过均值方差处理归一化图片，生成图片二进制文件
├── ReadMe.md
├── seresnext50_pth2onnx.py             //用于转换pth模型文件到onnx模型文件
└── vision_metric_ImageNet.py           //验证推理结果脚本，比对benchmark输出的分类结果和标签，给出Accuracy
```
## 4. 准备数据集
### 步骤1 获取原始数据集
本模型支持ImageNet 50000张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。
```text
├── imageNet
├── val
```
### 步骤2 数据预处理
预处理将原始数据集转换为模型输入的数据。

 二进制输入
```text
将原始数据（.jpeg）转化为二进制文件（.bin）。转化方法参考Trochvision训练预处理方法处理数据，以获得最佳精度。通过缩放、均值方差手段归一化，输出为二进制文件。
执行preprocess_se_resnext50_32x4d_pth.py脚本:

python3.7 preprocess_se_resnext50_32x4d_pth.py /opt/npu/imageNet/val ./prep_bin

第一个参数为原始数据验证集（.jpeg）所在路径，第二个参数为输出的二进制文件（.bin）所在路径。每个图像对应生成一个二进制文件。
```

### 步骤 3 生成数据集info文件

####  二进制输入info文件生成
```text
使用benchmark推理需要输入二进制数据集的info文件，用于获取数据集。使用get_info.py脚本，输入已经得到的二进制文件，输出生成二进制数据集的info文件。运行get_info.py脚本。
python3.7 get_info.py bin ./prep_bin ./seresnext50_val.info 224 224
第一个参数为生成的数据集文件格式，第二个参数为预处理后的数据文件路径，第三个参数为生成的数据集文件保存的路径，第四个和第五个参数分别为模型输入的宽度和高度。
运行成功后，在当前目录中生成seresnext50_val.info。
```

## 5. 模型推理
### 步骤1 模型转换
本模型基于开源框架PyTorch训练的SE_ResNeXt50_32x4d进行模型转换。
使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。
1. 获取权重文件方法。
```text
− 从源码中获取se_resnext50_32x4d-a260b3a4.pth文件
− 根据 http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth 
在PyTorch开源框架中获取se_resnext50_32x4d-a260b3a4.pth文件。
```
2.	导出.onnx文件。
```text
将模型权重文件.pt转换为.onnx文件。
a. 下载代码仓。
git clone https://github.com/Cadene/pretrained-models.pytorch.git (commit_id:8aae3d8f1135b6b13fed79c1d431e3449fdbf6e0)
b. 将代码仓上传至服务器任意路径下如（如：/home/HwHiAiUser）。
c. 进入代码仓目录并将seresnext50_pth2onnx.py和se_resnext50_32x4d-a260b3a4.pth移到pretrained-models.pytorch上级目录。
d. 进入pretrained-models.pytorch目录下，执行seresnext50_pth2onnx.py脚本将.pth文件转换为.onnx文件，执行如下命令。
python3.7 seresnext50_pth2onnx.py ../se_resnext50_32x4d-a260b3a4.pth ../se_resnext50_32x4d.onnx
第一个参数为输入权重文件路径，第二个参数为输出onnx文件路径。
运行成功后，在当前目录生成se_resnext50_32x4d.onnx模型文件。然后将生成onnx文件移到源码包中。
```

3.	使用ATC工具将ONNX模型转OM模型。
```shell
# 配置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 使用二进制输入时，执行如下命令
atc --model=./se_resnext50_32x4d.onnx --framework=5 --output=seresnext50_32x4d_16 --input_format=NCHW --input_shape="image:16,3,224,224" --log=info --soc_version=Ascend${chip_name}

# 参数说明
--model：为ONNX模型文件。
--framework：5代表ONNX模型。
--output：输出的OM模型。
--input_format：输入数据的格式。
--input_shape：输入数据的shape。
--log：日志等级。
--soc_version：部署芯片类型。
--chip_name: 部署芯片类型，利用 npu-smi info 查看芯片类型
```

### 步骤 2 开始推理验证。
1. 使用ais_infer工具进行推理
```shell
python3 ais_infer.py –model seresnext50_32x4d.om --input prep_bin/ --output ./ --outfmt TXT --batchsize {batch_size}
```
2. 精度验证
```shell
# 调用ais_verify.py脚本与数据集标签val_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。
python3.7 vision_metric_ImageNet.py ${ais_output} ./val_label.txt ./ result.json
# 第一个参数为生成推理结果所在路径，第二个参数为标签数据，第三个参数为生成结果文件。
```


## 6. 结果
### 1. 精度

|           |  310  | 310P  |
|:---------:|:-----:|:------|
| bs1 Top1  | 79.05 | 79.06 |
| 最优bs Top1 | 79.06 | 79.06 |
| bs1 Top5  | 94.44 | 94.44 |
| 最优bs Top5 | 94.44  | 94.44 |

### 2. 吞吐量
|  bs  |  310   | 310P      | T4        | 310P/310 | 310P/T4 |
|:----:|:------:|:----------|:----------|:---------|:--------|
| bs1	 |786.26	 | 1006.47   | 549.747   | 1.2801   | 1.8308  |
| bs4	 | 941.04 | 1804.86   | 750.772   | 1.9179   | 2.4040  |
| bs8	 |1030.82 | 1507.69   | 894.025   | 1.4626	  | 1.6864  |
| bs16 |1113.52 | 1396.36   | 957.024   | 1.2540   | 1.4591  |
| bs32 |995.492 | 600.3     | 1035.980	 | 0.603    | 	0.579  |
| bs64 | 953.8  | 1198.41   | 1054.064  | 1.2565   | 	1.1369 |
| 最优bs | 1113.52| 	1804.86	 | 1054.064  | 	1.6208  | 	1.7123 |
```text
最优batch： 310P 大于310的1.2倍；310P大于T4的1.6倍，性能达标
```