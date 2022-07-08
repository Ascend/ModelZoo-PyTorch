# Se_resnext50_32x4d

## ImageNet training with PyTorch

This implements training of Se_resnext50_32x4d on the ImageNet dataset, mainly modified from [Github](https://github.com/pytorch/examples/tree/master/imagenet)

## Se_resnext50_32x4d Detail
Can see in [Github](https://github.com/Cadene/pretrained-models.pytorch).

## Requirements
- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

## 1p training 1p
```
bash ./test/train_full_1p.sh  --data_path=数据集路径    # 精度训练
bash ./test/train_performance_1p.sh  --data_path=数据集路径  # 性能训练
```

## 8p training 8p
```
bash ./test/train_full_8p.sh  --data_path=数据集路径         # 精度训练
bash ./test/train_performance_8p.sh  --data_path=数据集路径  # 性能训练
```

## eval default 8p， should support 1p
`bash ./test/train_eval_8p.sh  --data_path=数据集路径`

## online inference demo
`python3.7.5 demo.py`

## To ONNX
`python3.7.5 pthtar2onx.py`
        

## Se_resnext50_32x4d training result

| Acc@1    | FPS       | Npu_nums| Epochs   | Type     |
| :------: | :------:  | :------ | :------: | :------: |
|     |  582      | 1       |   1    | O2       |
|  78.239  |  2953     | 8       | 100      | O2       |


## 准备数据集
### 步骤 1	获取原始数据集。
本模型支持ImageNet 50000张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。
```text
├── imageNet
├── val
```
### 步骤 2	数据预处理。
预处理将原始数据集转换为模型输入的数据。模型输入数据有两种格式，分别为二进制输入和jpeg图像输入。
1. 二进制输入
```text
将原始数据（.jpeg）转化为二进制文件（.bin）。转化方法参考Trochvision训练预处理方法处理数据，以获得最佳精度。通过缩放、均值方差手段归一化，输出为二进制文件。
执行preprocess_se_resnext50_32x4d_pth.py脚本:

python3.7 preprocess_se_resnext50_32x4d_pth.py /opt/npu/imageNet/val ./prep_bin

第一个参数为原始数据验证集（.jpeg）所在路径，第二个参数为输出的二进制文件（.bin）所在路径。每个图像对应生成一个二进制文件。
```
2. JPEG图片输入
```text
以JPEG图片作为输入时，需准备输入数据及AIPP配置文件。
a. 输入数据为原始数据（.jpeg）文件。
b. AIPP配置文件aipp_se_resnext50_32x4d_pth.config在源码包中已经提供。
输入数据通过DVPP工具实现解码、缩放并输出YUV数据，再通过AIPP进行色域转换及裁剪，最后输入网络模型中进行推理，方便快捷。AIPP配置文件在ATC工具进行模型转换的过程中插入AIPP算子，即可与DVPP处理后的数据无缝对接。但该输入方式和二进制输入方式相比模型精度下降了0.7%左右。
本模型使用Benchmark集成的DVPP，若用户需自己配置AIPP文件请参见以下指导。
DVPP使用指导请参见《CANN V100R020C20 应用软件开发指南 (C&C++)》。
AIPP参数配置请参见《CANN V100R020C20 开发辅助工具指南 (推理)》。
```
### 步骤 3	生成数据集info文件。

#### 1. 二进制输入info文件生成
```text
使用benchmark推理需要输入二进制数据集的info文件，用于获取数据集。使用get_info.py脚本，输入已经得到的二进制文件，输出生成二进制数据集的info文件。运行get_info.py脚本。
python3.7 get_info.py bin ./prep_bin ./seresnext50_val.info 224 224
第一个参数为生成的数据集文件格式，第二个参数为预处理后的数据文件路径，第三个参数为生成的数据集文件保存的路径，第四个和第五个参数分别为模型输入的宽度和高度。
运行成功后，在当前目录中生成seresnext50_val.info。
```
#### 2. JPEG图片输入info文件生成
```text
使用benchmark推理需要输入图片数据集的info文件，用于获取数据集。使用get_info.py脚本，输入已经获得的图片文件，输出生成图片数据集的info文件。运行get_info.py脚本。
python3.7 get_info.py jpg ../dataset/ImageNet/ILSVRC2012_img_val ./ImageNet.info
第一个参数为生成的数据集文件格式，第二个参数为预处理后的数据文件的相对路径，第三个参数为生成的数据集文件保存的路径。
运行成功后，在当前目录中生成seresnext50_val.info。
```
## 模型推理
### a． 310：
### 步骤 1	模型转换。
本模型基于开源框架PyTorch训练的SE_ResNeXt50_32x4d进行模型转换。
使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。
1. 两种获取权重文件方法。
```text
− 从源码中获取se_resnext50_32x4d-a260b3a4.pth文件
− 单击Link在PyTorch开源框架中获取se_resnext50_32x4d-a260b3a4.pth文件。
```
2.	导出.onnx文件。
```text
将模型权重文件.pt转换为.onnx文件。
a. 下载代码仓。
git clone https://github.com/Cadene/pretrained-models.pytorch.git
b. 将代码仓上传至服务器任意路径下如（如：/home/HwHiAiUser）。
c. 进入代码仓目录并将se_resnext50_32x4d-a260b3a4.pth与seresnext50_pth2onnx.py移到pretrained-models.pytorch-master/pretrainedmodels/models/目录下。
d. 进入models目录下，执行seresnext50_pth2onnx.py脚本将.pth文件转换为.onnx文件，执行如下命令。
python3.7 seresnext50_pth2onnx.py ./se_resnext50_32x4d-a260b3a4.pth ./se_resnext50_32x4d.onnx
第一个参数为输入权重文件路径，第二个参数为输出onnx文件路径。
运行成功后，在当前目录生成se_resnext50_32x4d.onnx模型文件。然后将生成onnx文件移到源码包中。
 
使用ATC工具将.onnx文件转换为.om文件，需要.onnx算子版本需为11。在seresnext50_pth2onnx.py脚本中torch.onnx.export方法中的输入参数opset_version的值需为11，请勿修改。
```

3.	使用ATC工具将ONNX模型转OM模型。
- 修改se_resnext50_32x4d_atc.sh脚本，通过ATC工具使用脚本完成转换，具体的脚本示例如下：
```shell
# 配置环境变量
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

# 使用二进制输入时，执行如下命令
atc --model=./se_resnext50_32x4d.onnx --framework=5 --output=seresnext50_32x4d_16 --input_format=NCHW --input_shape="image:16,3,224,224" --log=info --soc_version= Ascend710

# 使用JPEG输入时，执行如下命令
atc --model=./se_resnext50_32x4d.onnx --framework=5 --output=seresnext50_32x4d_aipp_16 --input_format=NCHW --input_shape="image:16,3,224,224" --log=info --insert_op_conf=aipp_TorchVision.config -soc_version=Ascend710 

# 参数说明
--model：为ONNX模型文件。
--framework：5代表ONNX模型。
--output：输出的OM模型。
--input_format：输入数据的格式。
--input_shape：输入数据的shape。
--log：日志等级。
--soc_version：部署芯片类型。
--insert_op_conf=aipp_TorchVision.config: AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用，详见下文数据预处理。
−	执行atc转换脚本，将.onnx文件转为离线推理模型文件.om文件。
bash SE_ResNeXt50_32x4d_atc.sh
运行成功后生成seresnext50_32x4d_16.om用于二进制输入推理的模型文件，生成的seresnext50_32x4d_aipp_16.om用于图片输入推理的模型文件。
```

### 步骤 2	开始推理验证。
1. 使用Benchmark工具进行推理。
```shell
# 增加执行权限
chmod u+x benchmark.x86_64
# 二进制输入执行
./benchmark.x86_64 -model_type=vision -om_path=seresnext50_32x4d_16.om -device_id=0 -batch_size=16 -input_text_path=seresnext50_val.info -input_width=224 -input_height=224 -output_binary=false -useDvpp=false
seresnext50_val.info为处理后的数据集信息。
# 图片输入执行
./benchmark.x86_64 -model_type=vision -om_path=seresnext50_32x4d_aipp_16.om -device_id=0 -batch_size=16 -input_text_path=ImageNet.info -input_width=256 -input_height=256 -useDvpp=true -output_binary=false
# ImageNet.info为图片信息。输入参数中的“input_height”和“input_weight”与AIPP节点输入一致，值为256因为AIPP中做了裁剪。benchmark.{arch}请根据运行环境架构选择，如运行环境为x86_64，需执行./benchmark.x86_64。参数详情请参见《CANN V100R020C20 推理benchmark工具用户指南》。推理后的输出默认在当前目录result下。
```

2. 精度验证。
```shell
# 调用vision_metric_ImageNet.py脚本与数据集标签val_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。
python3.7 vision_metric_ImageNet.py result/dumpOutput_device0/ ./val_label.txt ./ result.json
# 第一个参数为生成推理结果所在路径，第二个参数为标签数据，第三个参数为生成结果文件。
```

 ### b． 310P：
推理验证：
```shell
python3.7 vision_metric_ImageNet.py result/dumpOutput_device0/ ./val_label.txt ./ result.json
```


## 结果
### 1. 精度

|           |  310  | 710 |
|:---------:|:-----:|:----|
| bs1 Top1  | 79.05 | 79.06   |
| 最优bs Top1 | 79.06 | 79.06   |
| bs1 Top5  | 94.44 | 94.44   |
| 最优bs Top5 | 94.44  | 94.44   |

### 2. 吞吐量
|  bs  |  310   | 710       | T4        | 710/310 | 710/T4  |
|:----:|:------:|:----------|:----------|:--------|:--------|
| bs1	 |786.26	 | 1006.47   | 549.747   | 1.2801  | 1.8308  |
| bs4	 | 941.04 | 1804.86   | 750.772   | 1.9179  | 2.4040  |
| bs8	 |1030.82 | 1507.69   | 894.025   | 1.4626	 | 1.6864  |
| bs16 |1113.52 | 1396.36   | 957.024   | 1.2540  | 1.4591  |
| bs32 |995.492 | 2001.78   | 1035.980	 | 2.0108  | 	1.9323 |
| bs64 | 953.8  | 1198.41   | 1054.064  | 1.2565  | 	1.1369 |
| 最优bs | 1113.52| 	2001.78	 | 1054.064  | 	1.7977 | 	1.8991 |
```text
最优atch： 710 大于310的1.2倍；710大于T4的1.6倍，性能达标
```