### OCRNet模型Pytorch离线推理指导

### 1. 环境准备

#### 1.1 深度学习框架

```
CANN == 5.0.2
torch == 1.11.0
torchvision == 0.12.0
onnx 
```

#### 1.2 python第三方库

```sh
conda create -n ocrnet python==3.7
conda activate ocrnet
pip install -r requirements.txt
```

#### 1.3 paddlepaddle-gpu安装

paddlepaddle框架在linux系统上安装时，对cuda和cudnn版本的要求如下所示：

- CUDA toolkit 10.1/10.2 with cuDNN 7 (cuDNN version>=7.6.5)
- CUDA toolkit 11.0 with cuDNN v8.0.4
- CUDA toolkit 11.1 with cuDNN v8.1.1
- CUDA toolkit 11.2 with cuDNN v8.1.1

安装命令：

```sh
python -m pip install paddlepaddle-gpu==2.2.2 -i https://mirror.baidu.com/pypi/simple
```

#### 1.4 MagicONNX安装

```sh
git clone https://gitee.com/Ronnie_zheng/MagicONNX.git
cd MagicONNX
pip install .
```

#### 1.5 获取开源模型代码

```sh
git clone https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
pip install -r requirements.txt # 安装相关依赖
```

#### 1.6 获取模型权重文件

权重文件获取地址：https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.1/configs/ocrnet HRNet_w18 CityScapes 

#### 1.7 准备数据集

用户需自行获取CityScapes数据集中的验证集部分（无需训练集），上传数据集到服务器中，必须要与preprocess.py同目录。其内容如下所示：

```
cityscapes
    ├─gtFine
    │  └─val
    │      ├─frankfurt
    │      │      frankfurt_000000_000294_gtFine_color.png
    │      │      frankfurt_000000_000294_gtFine_instanceIds.png
    │      │      frankfurt_000000_000294_gtFine_labelIds.png
    │      │      frankfurt_000000_000294_gtFine_labelTrainIds.png
    │      │      frankfurt_000000_000294_gtFine_polygons.json
    │      ├─lindau
    │      └─munster
    └─leftImg8bit
        └─val
            ├─frankfurt
            │      frankfurt_000000_000294_leftImg8bit.png
            ├─lindau
            │      lindau_000000_000019_leftImg8bit.png
            └─munster
                    munster_000000_000019_leftImg8bit.png
```

#### 1.8 获取msame工具

将msame文件放到当前工作目录

```
git clone https://gitee.com/ascend/tools.git
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub
cd tools/msame
./build.sh g++ out
cd out
mv main ../../../msame
```



### 2. 模型转换

将步骤1.6中得到的pdparams模型权重保存在当前目录后，调用pth2om.sh脚本，即可将模型转为batch_size=1,4,8,16的om模型：

```sh
./test/pth2om.sh
```

得到的om模型保存在om目录中。

### 3.数据集预处理

在当前工作目录下，执行以下命令：

```
python OCRNet_preprocess.py --src_path /opt/npu/cityscapes/ --bin_file_path bs1_bin --batch_size 1
```

参数解释如下：

- src_path：数据集所在目录
- bin_file_path：预处理结果输出目录
- batch_size：将预处理结果输出的batch size

### 4.模型推理

在当前目录下对om模型执行推理，其中--model为之前转化好的bs为1的om模型，--input为步骤3中得到的预处理结果所在目录，--output为推理结果输出目录

```
./msame --model "/home/lzh/om_model/final/ocrnet_bs1.om" --input "/home/lzh/bs1_bin/imgs" --output "/home/lzh/out/" --outfmt TXT 
```

### 5.计算推理精度

使用后处理脚本，计算得到模型的推理精度：

```
python OCRNet_postprocess.py --bin_file_path bs1_bin --pred_path /home/lzh/out/202262_18_19_23_73731
```

参数解释如下：

- bin_file_path：步骤3得到的预处理结果所在目录
- pred_path：推理结果输出目录

### 6.精度对比

以下为测试出的batch_size=1的精度对比：

|      | 原模型精度(mIoU) | om模型精度(mIoU) |
| ---- | ---------------- | ---------------- |
| bs1  | 80.67%           | 79.89%           |

精度下降不超过百分之一，精度达标

### 7.性能对比

#### 7.1 GPU性能数据

**注意：**

> 测试gpu性能要确保device空闲，使用nvidia-smi命令可查看device是否在运行其它推理任务

以bs=1为例，这里的ocrnet_bs1.onnx为优化前onnx模型

```
trtexec --onnx=ocrnet_bs1.onnx --fp16 --threads --workspace=15000
```

#### 7.2 NPU性能数据

以bs=1为例，利用msame进行纯推理：

```
./msame --model "om/ocrnet_optimize_bs1.om.om" --output "result/" --outfmt TXT  --loop 100
```

| Model      | batch_size | T4Throughput/Card | 710Throughput/Card |
| ---------- | ---------- | ----------------- | ------------------ |
| ECAPA-TDNN | 1          | 11.918            | 15.223             |
| ECAPA-TDNN | 4          | 11.971            | 11.116             |
| ECAPA-TDNN | 8          | 11.980            | 9.353              |
| ECAPA-TDNN | 16         | /                 | 9.413              |

（batch_size=16时，T4运行OCRNet模型会出现算子输入shape超过上限的报错，因此忽略此情况下的推理）

15.223 / 11.918 = 1.28，性能达标

