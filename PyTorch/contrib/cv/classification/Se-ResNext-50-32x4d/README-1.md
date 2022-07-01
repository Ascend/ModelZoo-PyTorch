## 推理
### 1. 准备数据
#### a. 数据预处理：
```shell
# 将原始数据（.jpeg）转化为二进制文件（.bin）。转化方法参考Trochvision训练预处理方法处理数据，以获得最佳精度。通过缩放、均值方差手段归一化，输出为二进制文件。
python3.7 preprocess_se_resnext50_32x4d_pth.py /opt/npu/imageNet/val ./prep_bin
```
#### b. 生成数据集info文件
```shell
# 使用benchmark推理需要输入二进制数据集的info文件，用于获取数据集。使用get_info.py脚本，输入已经得到的二进制文件，输出生成二进制数据集的info文件。运行get_info.py脚本
python3.7 get_info.py bin ./prep_bin ./seresnext50_val.info 224 224
```
### 2. 模型推理
#### a. 模型转换
使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件
##### 1) 导出onnx文件
```text
1.下载代码仓
git clone https://github.com/Cadene/pretrained-models.pytorch.git

2.进入代码仓目录并将se_resnext50_32x4d-a260b3a4.pth与seresnext50_pth2onnx.py移到pretrained-models.pytorch-master/pretrainedmodels/models/目录下

3.进入models目录下，执行seresnext50_pth2onnx.py脚本将.pth文件转换为.onnx文件，执行如下命令:
python3.7 seresnext50_pth2onnx.py ./se_resnext50_32x4d-a260b3a4.pth ./se_resnext50_32x4d.onnx

4.运行成功后，在当前目录生成se_resnext50_32x4d.onnx模型文件。然后将生成onnx文件移到源码包中。
```
##### 2) 使用ATC工具将ONNX模型转OM模型
修改se_resnext50_32x4d_atc.sh脚本，通过ATC工具使用脚本完成转换，具体的脚本示例如下：
```text
# 配置环境变量
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

# 使用二进制输入时，执行如下命令
atc --model=./se_resnext50_32x4d.onnx --framework=5 --output=seresnext50_32x4d_16 --input_format=NCHW --input_shape="image:16,3,224,224" --log=info --soc_version=Ascend310

# 使用JPEG输入时，执行如下命令
atc --model=./se_resnext50_32x4d.onnx --framework=5 --output=seresnext50_32x4d_aipp_16 --input_format=NCHW --input_shape="image:16,3,224,224" --log=info --insert_op_conf=aipp_TorchVision.config -soc_version=Ascend310
```
执行atc转换脚本，将.onnx文件转为离线推理模型文件.om文件。
```text
bash SE_ResNeXt50_32x4d_atc.sh
```
#### a. 开始推理
#### 1) 使用Benchmark工具进行推理
```text
1.增加benchmark.{arch}可执行权限
chmod u+x benchmark.x86_64

2.二进制输入
./benchmark.x86_64 -model_type=vision -om_path=seresnext50_32x4d_16.om -device_id=0 -batch_size=16 -input_text_path=seresnext50_val.info -input_width=224 -input_height=224 -output_binary=false -useDvpp=false
```
#### 2) 精度验证
```text
调用vision_metric_ImageNet.py脚本与数据集标签val_label.txt比对，可以获得Accuracy数据，结果保存在result.json中
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
|  bs  |  310   | 710     |T4| 710/310     |710/T4|
|:----:|:------:|:--------|:----|:------------|:----|
| bs1	 |786.26	 | 969.803 |549.7465668| 1.233438048 |1.764091053|
| bs4	 | 941.04 | 1735.71 |750.7718873| 1.844459322 |2.311900631|
| bs8	 |1030.82 | 1456.65 |894.0245633| 1.41309831	 |1.629317649|
| bs16 |1113.52 | 1355.26 |957.0236564| 1.217095337 |1.416119644|
| bs32 |995.492 | 607.488 |1035.980912	| 0.610238957 |	0.586389182|
| bs64 | 953.8  | 1190.84 | 622.4893739| 1.248521703 |	1.913028639|
| 最优bs | 1113.52|	1735.71	|1035.980912|	1.558759609|	1.675426622|
```text
最优atch： 710 大于310的1.2倍；710大于T4的1.6倍，性能达标
```