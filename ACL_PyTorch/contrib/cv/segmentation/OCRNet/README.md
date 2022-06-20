### OCRNet模型Pytorch离线推理指导

**注**：在下文的叙述中，“当前工作目录”均指README.md所在目录。

### 1. 环境准备

#### 1.1 深度学习框架

```
CANN == 5.1.RC1
paddlepaddle == 2.3.0
onnx 
```

#### 1.2 python第三方库

```sh
conda create -n ocrnet python==3.7
conda activate ocrnet
pip install -r requirements.txt
```

#### 1.3 MagicONNX安装

```sh
git clone https://gitee.com/Ronnie_zheng/MagicONNX.git
cd MagicONNX
pip install .
```

#### 1.4 获取开源模型代码

```sh
git clone https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
git checkout remotes/origin/release/2.1
pip install -r requirements.txt # 安装相关依赖
```

#### 1.5 获取模型权重文件

可通过以下链接获取权重文件：

[OCRNet权重文件下载链接](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/ocrnet_hrnetw18_cityscapes_1024x512_160k/model.pdparams)

#### 1.6 准备数据集

用户需自行获取CityScapes数据集中的验证集部分（无需训练集），上传数据集到服务器中。其内容如下所示：

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

#### 1.7 获取msame工具

将msame文件放到当前工作目录。

```sh
git clone https://gitee.com/ascend/tools.git
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub
cd tools/msame
./build.sh g++ out
cd out
mv main ../../../msame
```



### 2. 模型转换

将步骤1.5中得到的pdparams模型权重保存在当前工作目录后，调用pth2om.sh脚本，即可将模型转为batch_size=1,4,8,16的om模型：

```sh
./test/pth2om.sh Ascend${chip_name} # Ascend310P3
```

其中${chip_name}可通过`npu-smi info`指令查看：

![Image](https://gitee.com/Ronnie_zheng/ascend-pytorch-crowdintelligence-doc/raw/master/Ascend-PyTorch%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%8C%87%E5%AF%BC/images/310P3.png)

得到的om模型保存在om目录中。

### 3.数据集预处理

在当前工作目录下，执行以下命令：

```sh
python3 OCRNet_preprocess.py --src_path /opt/npu/cityscapes/ --bin_file_path bs1_bin --batch_size 1
```

参数解释如下：

- src_path：数据集所在目录
- bin_file_path：预处理结果输出目录
- batch_size：将预处理结果输出的batch size

### 4.模型推理

在当前工作目录下对om模型执行推理，其中--model为之前转化好的bs为1的om模型，--input为步骤3中得到的预处理结果所在目录，--output为推理结果输出目录。

```sh
./msame --model "/home/lzh/om_model/final/ocrnet_bs1.om" --input "/home/lzh/bs1_bin/imgs" --output "/home/lzh/out/" --outfmt TXT 
```

### 5.计算推理精度

在当前工作目录下，使用后处理脚本，计算得到模型的推理精度：

```sh
python3 OCRNet_postprocess.py --bin_file_path bs1_bin --pred_path /home/lzh/out
```

参数解释如下：

- bin_file_path：步骤3得到的预处理结果所在目录
- pred_path：推理结果保存目录

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

```sh
trtexec --onnx=ocrnet_bs1.onnx --fp16 --threads --workspace=15000
```

#### 7.2 NPU性能数据

以bs=1为例，利用msame进行纯推理：

```sh
./msame --model "om/ocrnet_optimize_bs1.om.om" --output "result/" --outfmt TXT  --loop 100
```

| Model  | batch_size | 基准性能 | 310P性能 |
| ------ | ---------- | ----------------- | ------------------ |
| OCRNet | 1          | 11.918            | 15.223             |
| OCRNet | 4          | 11.971            | 11.116             |
| OCRNet | 8          | 11.980            | 9.353              |
| OCRNet | 16         | /                 | 9.413              |

（batch_size=16时，T4运行OCRNet模型会出现算子输入shape超过上限的报错，因此忽略此情况下的推理）

15.223 / 11.918 = 1.28

