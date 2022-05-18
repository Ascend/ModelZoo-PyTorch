# OpenPose模型PyTorch离线推理指导

## 1 模型概述
### 1.1 参考论文
[Lightweight OpenPose论文](https://arxiv.org/abs/1811.12004)
### 1.2 参考实现
[代码地址](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)
> branch: master

> commit id: 1590929b601535def07ead5522f05e5096c1b6ac

## 2 环境准备

### 2.1 环境介绍
CANN=[5.1RC1](https://www.hiascend.com/software/cann/commercial?version=5.1RC1)。 
硬件环境、开发环境和运行环境准备请参见[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/51RC1/envdeployment/instg)
### 2.2 所需依赖
```
torch==1.5.0
torchvision=0.6.0
onnx==1.7.0
pycocotools==2.0.4
opencv-python==4.5.2.52
numpy==1.21.6
pillow==7.2.0
```
### 2.3 安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装
```
pip3.7 install -r requirements.txt
```
### 3 数据准备
### 3.1 获取数据集  
服务器上可能已经下载好该数据集，若无，参考以下方法下载。
[coco2017官网](https://cocodataset.org/#download)  
下载其中val2017图片及其标注文件，使用5000张验证集进行测试，图片与标注文件分别存放在/root/datasets/coco/val2017与/root/datasets/coco/annotations/person_keypoints_val2017.json。
文件目录结构如下，
```
root
├── datasets
│   ├── coco
│   │   ├── annotations
│   │   │   ├── captions_train2017.json
│   │   │   ├── captions_val2017.json
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   │   ├── person_keypoints_train2017.json
│   │   │   └── person_keypoints_val2017.json
│   │   ├── val2017
│   │   ├── annotations_trainval2017.zip
```
### 3.2 数据预处理
a 建立数据存储文件夹
```
mkdir  ./datasets/coco/prep_dataset
mkdir ./output
```
b 生成bin文件
```
python3.7 OpenPose_preprocess.py --src_path ${DATASET_PATH} --save_path ./datasets/coco/prep_dataset --pad_txt_path ./output/pad.txt
```
参数说明：
- --src_path：为数据集路径。
- --save_path：为模型输入数据存储路径。
- --pad_txt_path：输出文件路径。
### 3.3 生成数据集info文件
```
python3.7 gen_dataset_info.py bin  ./datasets/coco/prep_dataset ./openpose_prep_bin.info 640 368
```
## 4 模型转换
### 4.1 获取源码
```
git clone https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch.git  
cd lightweight-human-pose-estimation.pytorch
git checkout master
git reset --hard 1590929b601535def07ead5522f05e5096c1b6ac
cd -
```
### 4.2 获取权重文件  
[OpenPose预训练pth权重文件](https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth)
```
wget https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth -P ./weights
```
### 4.3 pth转onnx模型
```
python3.7 OpenPose_pth2onnx.py --checkpoint_path='./checkpoint_iter_370000.pth' --output_name="./human-pose-estimation.onnx"
```
### 4.4 onnx转om模型
设置环境变量
```
source /usr/local/Ascend/ascend-lastest/set_env.sh
```
使用ATC工具转换，工具使用方法可以参考[《CANN 开发辅助工具指南 (推理)》](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)
```
atc --framework=5 --model=./human-pose-estimation.onnx --output=./human-pose-estimation_bs1 --input_format=NCHW --input_shape="data:1, 3, 368, 640" --log=debug --soc_version=Ascend310
```
>  **说明**
> 注意目前ATC支持的onnx算子版本为11
## 5 推理验证
### 5.1 获取benchmark工具
[下载](https://gitee.com/ascend/cann-benchmark/tree/master/infer)
将benchmark.x86_64或benchmark.aarch64放到当前目录，并更改权限
```
chmod 777 benchmark.x86_64
```
### 5.2 离线推理
310/710上执行，执行时使npu-smi info查看设备状态，确保device空闲
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./human-pose-estimation_bs1.om -input_text_path=./openpose_prep_bin.info -input_width=640 -input_height=368 -output_binary=False -useDvpp=False
```
运行该指令后输出结果默认保存在当前目录/result/dumpOutput_device0中，同时在/result目录下会生成一个推理性能文件
### 5.3 获取性能信息
```
tail result/perf_vision_batchsize_1_device_0.txt
```
运行该指令获得bs为1时推理所得性能信息，实例如下：
```
[e2e] throughputRate: 0.0806233, latency: 6.20168e+07
[data read] throughputRate: 40.3632, moduleLatency: 24.775
[preprocess] throughputRate: 39.412, moduleLatency: 25.373
[inference] throughputRate: 32.3342, Interface throughputRate: 755.13, moduleLatency: 30.6295
[postprocess] throughputRate: 0.0806403, moduleLatency: 12400.8
```
### 5.4 gpu设备的推理
将onnx模型置于装有gpu的设备，参考以下指令获取onnx模型在gpu上推理的性能信息
```
trtexec --onnx=human-pose-estimation.onnx --fp16 --shapes=data:1x3x368x640
```
运行该指令获得bs为1时推理所得性能信息，实例如下：
```
[I] GPU Compute Time: 
min = 3.56421 ms, 
max = 6.29639 ms, 
mean = 4.08213 ms, 
median = 4.04974 ms, 
percentile(99%) = 6.24855 ms
```
### 5.5 精度验证
参考以下指令生成精度信息文件，并生成精度信息
```
python3.7 OpenPose_postprocess.py --benchmark_result_path result/dumpOutput_device0/  --labels  ${LABLES_PATH}  --pad_txt_path ./output/pad.txt --detections_save_path ./output/result_b1.json
```
参数说明：
- --benchmark_result_path：生成推理结果所在路径。
- --labels ：标签数据。
- --pad_txt_path：填充信息。
- --detections_save_path：生成结果文件。

## 6 测评结果
| 模型            | Ascend310（samples/s） | Ascend710（samples/s） | T4（samples/s） | 710/310 | 710/T4 | 精度    |
|---------------|----------------------|----------------------|---------------|---------|--------|-------|
| OpenPose_bs1  | 547.996              | 674.06               | 252.592       | 1.23    | 2.66   | 0.404 |