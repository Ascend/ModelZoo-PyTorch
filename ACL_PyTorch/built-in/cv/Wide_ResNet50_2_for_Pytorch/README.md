# Wide_ResNet50_2 ONNX模型端到端推理指导

### 文件说明
1.aipp_wide_resnet50_2.config：数据集aipp预处理配置文件
2.get_info.py：  生成推理输入的数据集二进制info文件或jpg info文件
3.preprocess.py：数据集预处理脚本，通过均值方差处理归一化图片，生成图片二进制文件
4.pth2onnx.py:   用于转换pth模型文件到onnx模型文件
5.env.sh：       用于设定设备推理执行所需环境变量
6.vision_metric_ImageNet.py：验证推理结果脚本，比对benchmark输出的分类结果和标签，给出Accuracy


## 2 环境说明
CANN 5.0.1
pytorch >= 1.5.0
torchvision >= 0.6.0
onnx >= 1.7.0
numpy == 1.18.5
Pillow == 7.2.0
opencv-python == 4.2.0.34


## 3 模型转换

### 3.1 pth转onnx模型

1. 下载pth权重文件  

[wide_resnet50_2权重文件下载](https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth)

```
wget https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth
```

2. 下载模型代码
```
git clone https://github.com/pytorch/vision
cd vision
git reset 7d955df73fe0e9b47f7d6c77c699324b256fc41f --hard
python3.7 setup.py install
cd ..
```

3. 执行pth2onnx脚本，生成onnx模型文件
```python
python3.7 pth2onnx.py ./wide_resnet50_2-95faca4d.pth ./wide_resnet50_2.onnx
```
第一个参数为输入权重文件路径，第二个参数为输出onnx文件路径。运行成功后，在当前目录生成wide_resnet50_2.onnx模型文件
	
 **模型转换要点：**  
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明

### 3.2 onnx转om模型

1.设置环境变量

```
source env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考<<CANN 5.0.1 开发辅助工具指南 (推理) 01>>
使用二进制输入时，执行如下命令
```
atc --framework=5 --model=wide_resnet50_2.onnx --output=wide_resnet50_2_bs1 --input_format=NCHW --input_shape="image:1,3,224,224" --log=error --soc_version=Ascend310
```
	
使用JPEG输入时，执行如下命令
```
atc --framework=5 --model=wide_resnet50_2.onnx --output=wide_resnet50_2_dvpp_bs1 --input_format=NCHW --input_shape="image:1,3,224,224" --insert_op_conf=aipp_wide_resnet50_2  --log=error --soc_version=Ascend310
```


## 4 数据集预处理

### 4.1 数据集获取
该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/opt/npu/imagenet/val与/opt/npu/imagenet/val_label.txt

### 4.2 数据集预处理

1.执行预处理脚本preprocess.py，生成数据集预处理后的bin文件

```
python3.7 preprocess.py /home/HwHiAiUser/dataset/ImageNet/ILSVRC2012_img_val ./prep_bin
```
### 4.3 生成数据集信息文件

1.执行生成数据集信息脚本get_info.py，生成数据集信息文件

二进制输入info文件生成
```
python3.7 get_info.py bin ./prep_bin ./wide_resnet50_2_prep_bin.info 224 224
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息


图片输入info文件生成
```
python3.7 get_info.py jpg ../dataset/ImageNet/ILSVRC2012_img_val ./ImageNet.info
```
第一个参数为模型输入的类型，第二个参数为预处理后的数据文件相对路径，第三个为输出的info文件


## 5 离线推理

1.设置环境变量

```
source env.sh
```

2.增加benchmark可执行权限

```
chmod u+v benchmark.x86_64
```

3. 执行离线推理

二进制类型输入推理命令
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=wide_resnet50_2_bs1.om -input_text_path=./wide_resnet50_2_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```


图片输入推理命令
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=wide_resnet50_2_dvpp_bs1.om -input_text_path=./ImageNet.info -input_width=256 -input_height=256 -output_binary=False -useDvpp=true
```
ImageNet.info为图片信息。输入参数中的“input_height”和“input_weight” 与AIPP节点输入一致，值为256因为AIPP中做了裁剪


## 6 离线推理精度统计

后处理统计TopN精度

调用vision_metric_ImageNet.py脚本推理结果与label比对，可以获得Accuracy Top1&Top5数据，结果保存在result.json中。
```
python3.7 vision_metric_ImageNet.py result/dumpOutput_device0/ ./val_label.txt ./ result.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  
查看输出结果：