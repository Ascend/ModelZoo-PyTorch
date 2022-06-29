## 1、环境准备

### 1.1 下载pth模型
下载[pth模型](https://pan.baidu.com/s/1xGZFEQTjhsP4sj0Lvf_sXg)到推理目录下
提取码: e5ci 

### 1.2 下载onn_tools用于模型优化
```
git clone https://gitee.com/zheng-wengang1/onnx_tools.git
cd onnx_tools
git reset cbb099e5f2cef3d76c7630bffe0ee8250b03d921 --hard
cd ..
```

### 1.3 安装必要环境
```
pip install -r requirements.txt  
```

### 1.4 获取,安装,修改开源模型代码
```
git clone https://github.com/HRNet/HRNet-Semantic-Segmentation.git 
cd HRNet-Semantic-Segmentation
git reset 0bbb2880446ddff2d78f8dd7e8c4c610151d5a51 --hard
patch -p1 < ../HRNet.patch 
cd ..
```

### 1.5 下载数据集
下载[cityscpaes数据集](https://www.cityscapes-dataset.com)
解压后数据集目录结构如下:
```c
  ${data_path}
  |-- cityscapes
      |-- gtFine
      |    |-- test
      |    |-- train
      |    |-- val
      |-- leftImg8bit
           |-- test
           |-- train
           |-- val 
```

### 1.6 获取benchmark工具 
参考[CANN5.0.1 推理benchmark工具用户指南01](https://support.huawei.com/enterprise/zh/doc/EDOC1100191895?idPath=23710424%7C251366513%7C22892968%7C251168373)下载benchmark工具，并放置在推理目录下


## 2、离线推理
310上执行`npu-smi info`查看设备状态，确保device空闲      
### 2.1 模型转换

#### 2.1.1 生成onnx

执行如下指令会生成hrnet.onnx
```
python3 HRNet_pth2onnx.py --pth=hrnet.pth
```
优化模型，生成优化后的hrnet.onnx

```
python3 performance_optimize_resize.py hrnet.onnx hrnet.onnx
```

#### 2.1.2 生成om

配置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

使用atc指令将onnx转换为om

```
atc --framework=5 --model=hrnet.onnx --output=hrnet_bs1 --input_format=NCHW --input_shape="image:1,3,1024,2048" --log=debug --soc_version=Ascend${chip_name} --out_nodes="Conv_1380:0;Conv_1453:0"
```

注：${chip_name}可根据"npu-smi info"命令查看处理器型号

### 2.2 精度统计及性能

执行如下指令会在result目录下生成模型的推理结果文件以及推理性能文件,这里已经将数据集放置在/opt/npu/目录下:

执行“HRNet_preprocess.py”脚本，完成预处理。

```
mkdir prep_dataset
python3 HRNet_preprocess.py --src_path=${DATASET_PATH} --save_path=./prep_dataset
```
生成数据集info文件。

```
python3 gen_dataset_info.py bin ./prep_dataset ./hrnet_prep_bin.info 2048 1024
```

使用benchmark工具进行推理。

```
 ./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=hrnet_bs1.om -input_text_path=./hrnet_prep_bin.info -input_width=2048 -input_height=1024 -output_binary=True -useDvpp=False
```

推理性能文件参考如下:

```
[e2e] throughputRate: 2.05639, latency: 243144
[data read] throughputRate: 3.41281, moduleLatency: 293.013
[preprocess] throughputRate: 2.4649, moduleLatency: 405.695
[inference] throughputRate: 2.07674, Interface throughputRate: 7.59598, moduleLatency: 471.777
[postprocess] throughputRate: 2.08006, moduleLatency: 480.756
```
调用“HRNet_postprocess.py”脚本,可以获得Accuracy数据mean_Iou存储于“result_bs1.json”文件中。

```
python3 HRNet_postprocess.py result/dumpOutput_device0/ ${DATASET_PATH}/cityscapes/gtFine/val/ ./   result_bs1.json
```

精度结果如下：

```
IoU_array: [0.98409603 0.8683093  0.93306161 0.54274296 0.63750117 0.71053525
 0.7463551  0.82410723 0.93007527 0.63916374 0.95356734 0.84330032
 0.65674466 0.95762571 0.87352036 0.91712741 0.84777081 0.69376729
 0.79921667]
 mean_IoU: 0.8083467483255192
```
将onnx模型置于t4设备，获取onnx模型在gpu上推理的性能信息

```
trtexec --onnx=hrnet.onnx --fp16 --shapes=image:1x3x1024x2048 --threads
```

### 2.3 评测结果

|3D HRNet模型|  gpu吞吐率| 310吞吐率 | 310P吞吐率 |  目标精度| 310精度| 310P精度 |
|--|--|--|--|--|--|--|
|  bs1|  5.85fps| 4.90fps| 10.59fps |81.6%|80.85%|80.83%|
|  bs4| 5.75fps| 4.78fps| 8.06fps |81.6% |80.85%|80.83%|

注：该模型仅支持bs1和bs4