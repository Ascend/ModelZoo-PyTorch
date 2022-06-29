# FCOS模型PyTorch离线推理指导

## 1 环境准备 

### 1.1 安装必要的依赖

测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip install -r requirements.txt  
```
说明：PyTorch选用开源1.8.0版本

requirements.txt中的相关依赖如下：

```
torch==1.8.0
torchvision==0.9.0
onnx==1.9.0
numpy==1.21.1
opencv-python==4.5.2.52
```

另外还需要安装依赖

```
apt-get install libgl1-mesa-glx

pip install opencv-python-headless==4.1.2.30
```



### 1.2 获取，修改与安装开源模型代码  

安装mmcv  
{cuda_version}和{torch_version}根据自己的环境，参考https://github.com/open-mmlab/mmcv 的指导替换

```shell
git clone https://github.com/open-mmlab/mmcv
cd mmcv
MMCV_WITH_OPS=1 pip install -e .
cd ..
```
安装mmdetection
```
git clone https://github.com/open-mmlab/mmdetection -b master
cd mmdetection
git reset --hard dd0e8ede1f6aa2b65e8ce69826314b76751d4151
cp ../fcos_r50_caffe_fpn_4x4_1x_coco.py ./configs/fcos  
patch -p1 < ../fcos.diff
pip install -r requirements/build.txt
python setup.py develop
cd ..
```
通过命令找到pytorch安装位置。

```
pip show torch
```

返回pytorch安装位置（如：/root/anaconda3/envs/liuhuan/lib/python3.7/site-packages）。打开文件改路径下的/torch/onnx/utils.py文件。

搜索_check_onnx_proto(proto)并注释代码，添加pass代码，后保存并退出。

```
	if enable_onnx_checker and \
    	operator_export_type is OperatorExportTypes.ONNX and \
        	not val_use_external_data_format:
   	 # Only run checker if enabled and we are using ONNX export type and
    	# large model format export in not enabled.
    	# _check_onnx_proto(proto)
    	pass
```

### 1.3 获取权重文件  

[fcos基于mmdetection预训练的pth权重文件](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_r50_caffe_fpn_1x_4gpu_20190516-a7cac5ff.pth)  

### 1.4 [获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  

将benchmark.x86_64或benchmark.aarch64放到当前目录  

## 2准备数据集   

###   2.1 获取原始数据集

[coco2017](https://cocodataset.org/#download)，下载其中val2017图片及其标注文件，放入/opt/npu/coco文件夹（注：310p上已有此数据集），val2017目录存放coco数据集的验证集图片，annotations目录存放coco数据集的instances_val2017.json，文件目录结构如下：

```
opt
├── npu
│   ├── coco
│   │   ├── annotations
│   │   ├── val2017
```



### 2.2 数据预处理

执行命令，完成数据集预处理

```
python fcos_pth_preprocess.py --image_src_path=/opt/npu/coco/val2017 --bin_file_path=val2017_bin --model_input_height=800 --model_input_width=1333
```

### 2.3 生成数据集info文件

使用benchmark推理需要输入图片数据集的info文件，用于获取数据集。使用get_info.py脚本，输入已经获得的图片文件，输出生成图片数据集的info文件。运行get_info.py脚本。

#### 2.3.1 预处理后的二进制info文件生成

```
python get_info.py bin val2017_bin fcos.info 1333 800
```

第一个参数为生成的数据集文件格式，第二个参数为预处理后的数据文件的**相对路径**，第三个参数为生成的数据集文件保存的路径，第4，5个参数为图片的长宽。运行成功后，在当前目录中生成fcos.info。

#### 2.3.2 图片的info文件生成

```
python get_info.py jpg /opt/npu/coco/val2017 fcos_jpeg.info
```

## 3 离线推理 

### 3.1 模型转换

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

#### 3.1.1 获取权重文件。

在源码包中已经提供权重文件fcos_r50_caffe_fpn_1x_4gpu_20190516-a7cac5ff.pth。

#### 3.1.2 导出onnx文件

pth2onnx.py脚本将.pth文件转换为.onnx文件

```
python mmdetection/tools/deployment/pytorch2onnx.py mmdetection/configs/fcos/fcos_r50_caffe_fpn_4x4_1x_coco.py ./fcos_r50_caffe_fpn_1x_4gpu_20190516-a7cac5ff.pth --output-file fcos.onnx --input-img mmdetection/demo/demo.jpg --test-img mmdetection/tests/data/color.jpg --shape 800 1333 --dynamic-export
```

--dynamic-export参数是否确定导出动态batchsize的ONNX模型，如不添加为False。

使用ATC工具将.onnx文件转换为.om文件，导出.onnx模型文件时需设置算子版本为11。

#### 3.1.3 使用ATC工具将ONNX模型转OM模型

##### 3.1.3.1配置环境变量

source /usr/local/Ascend_5_0_4/ascend-toolkit/set_env.sh

此环境变量可能需要根据实际CANN安装路径修改

##### 3.1.3.2执行命令，将.onnx文件转为离线推理模型文件.om文件

${chip_name}可通过`npu-smi info`指令查看，例：310P3

```
atc --framework=5 --model=./fcos.onnx --output=fcos_bs1_310p --input_format=NCHW --input_shape="input:1,3,800,1333" --log=debug --soc_version=Ascend${chip_name}
```

- 参数说明：
  - --model：为ONNX模型文件。
  - --framework：5代表ONNX模型。
  - --output：输出的OM模型。
  - --input_format：输入数据的格式。
  - --input_shape：输入数据的shape。
  - --log：日志级别。
  - --soc_version：处理器型号，Ascend310或Ascend710。

运行成功后生成生成的fcos_bs1_310p.om用于图片输入推理的模型文件。

### 3.2 开始推理验证

#### 3.2.1 使用Benchmark工具进行推理

增加benchmark.*{arch}可执行权限*。

```
chmod u+x benchmark.x86_64
```

#### 3.2.2 执行推理

310p上执行时使npu-smi info查看设备状态，确保device空闲 

```
rm -rf result/dumpOutput_device0
./benchmark.x86_64 -model_type=vision -om_path=fcos_bs1_310p.om -device_id=0 -batch_size=1 -input_text_path=fcos.info -input_width=1333 -input_height=800 -useDvpp=false -output_binary=true
```

#### 3.2.3 数据后处理

调用fcos_pth_postprocess.py评测map精度。

```
python fcos_pth_postprocess.py --bin_data_path=./result/dumpOutput_device0/ --test_annotation=fcos_jpeg.info --det_results_path=./ret_npuinfer/ --net_out_num=3 --net_input_height=800 --net_input_width=1333 --ifShowDetObj --annotations_path /opt/npu/coco/annotations
```

#### 3.2.3 获取性能

执行parse.py获取性能

```
pythonpip test/parse.py result/perf_vision_batchsize_1_device_0.txt
```



 **评测结果：**   
| 模型      | 在线推理精度  | 310p离线推理精度 | 基准性能    | 310p性能   |
| :------: | :------: | :------: | :------:  | :------:  |
| FCOS bs1  | mAP：35.0% | mAP:35.9% |  5.1fps | 42.6019fps |
| FCOS bs4 | mAP：35.0% | mAP:35.9% | 5.4fps | 49.7825fps |

## 4 AOE调优

​	推理性能并不能达到性能要求，需要使用AOE工具进行算子调优

```
aoe --framework=5 --model=./fcos.onnx --job_type=2  --output=fcos_bs1_310p_aoe --input_format=NCHW --input_shape="input:1,3,800,1333" --device=0
```

参数说明：

- --model：为ONNX模型文件。
- --framework：5代表ONNX模型。
- --output：输出的OM模型。
- --input_format：输入数据的格式。
- --input_shape：输入数据的shape。
- --job_type：调优类型，1为子图调优，2为算子调优。子图调优在本模型上性能提升。

在调优过程中报错，查看/root/ascend/log/plog下相应的日志，发现报错："ModuleNotFoundError: No module named 'absl'"
	pip install absl-py 无法安装
	需要去官网下载[安装包](https://pypi.org/project/absl-py/#files)

​	上传到服务器解压
​	cd到absl-py-0.1.10目录下

	1.Python setup.py build
	
	2.Python setup.py install

即可解决问题
