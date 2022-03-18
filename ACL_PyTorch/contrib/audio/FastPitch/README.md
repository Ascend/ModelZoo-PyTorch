# FastPitch模型端到端推理指导

## 1 模型概述

- **[论文地址](https://arxiv.org/abs/2006.06873)**
- **[代码地址](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch)**

### 1.1 论文地址

[FastPitch论文](https://arxiv.org/abs/2006.06873)
Fastpitch模型由双向 Transformer 主干（也称为 Transformer 编码器）、音调预测器和持续时间预测器组成。 在通过第一组 N 个 Transformer 块、编码后，信号用基音信息增强并离散上采样。 然后它通过另一组 N个 Transformer 块，目的是平滑上采样信号，并构建梅尔谱图。

### 1.2 开源代码地址

[FastPitch开源代码](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch)

## 2 环境说明

### 2.1 深度学习框架

```
onnx==1.9.0
torch==1.8.0
```

### 2.2 python第三方库

```
matplotlib
numpy
inflect
librosa==0.8.0
scipy
Unidecode
praat-parselmouth==0.3.3
tensorboardX==2.0
dllogger
```

**说明：**

> X86架构：pytorch和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
> Arm架构：pytorch和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

### pth转om模型

1.下载pth权重文件
```
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/audio/FastPitch/pretrained_models.zip
```
（waveglow为语音生成器，不在本模型范围内, 但为了确保代码能正常运行，需要下载）

2.安装相关依赖

```
cd FastPitch
pip install -r requirements.txt
```

3.激活相关环境

```
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

3.pth转onnx, onnx简化，onnx转om。(以batch_size=1为例)

```
# 导出onnx
python pth2onnx.py -i phrases/tui_val100.tsv -o ./output/audio_tui_val100 --log-file ./output/audio_tui_val100/nvlog_infer.json --fastpitch pretrained_models/fastpitch/nvidia_fastpitch_210824.pt --waveglow pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt --wn-channels 256 --energy-conditioning --batch-size 1
# 简化onnx
python -m onnxsim ./test/models/FastPitch_bs1.onnx ./test/models/FastPitch_bs1_sim.onnx
# 转出om
atc --framework=5 --model=./test/models/FastPitch_bs1_sim.onnx --output=./test/models/FastPitch_bs1 --input_format=ND --input_shape="input:1,200" --out_nodes='Transpose_2044:0' --log=debug --soc_version=Ascend310
```

输出在/test/models中。



## 4 数据集预处理

### 4.1 数据集获取

（可选）本项目默认将数据集存放于/opt/npu/

```
cd ..
wget https://ascend-pytorch-one-datasets.obs.cn-north-4.myhuaweicloud.com/train/zip/LJSpeech-1.1.zip
unzip LJSpeech-1.1.zip
mv /LJSpeech-1.1 /opt/npu/
```

### 4.2 数据集预处理

生成输入数据，并准备输出标签和pth权重的输出数据。本模型的验证集大小为100，具体信息在phrases/tui_val100.tsv文件中。

- FastPitch模型的输入数据是由文字编码组成，输入长度不等，模型已经将其补零成固定长度200。将输入数据转换为bin文件方便后续推理，存入test/input_bin文件夹下，且生成生成数据集预处理后的bin文件以及相应的info文件。
- 在语音合成推理过程中，输出为mel图谱，本模型的输出维度为batch_size×900×80。将其输出tensor存为pth文件存入test/mel_tgt_pth文件夹下。
- 同时，为了后面推理结束后将推理精度与原模型pth权重精度进行对比，将输入数据在pth模型中前传得到的输出tensor村委pth文件存入test/mel_out_pth文件夹下。

以上步骤均执行下面指令完成：

```
python data_process.py -i phrases/tui_val100.tsv -o ./output/audio_tui_val100 --log-file ./output/audio_tui_val100/nvlog_infer.json --fastpitch pretrained_models/fastpitch/nvidia_fastpitch_210824.pt --waveglow pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt
```

## 5 离线推理及精度对比

### 5.1 使用benchmark工具推理

获取benchmark工具

### 5.2 模型推理

- 使用benchmark工具进行推理(以batch_size=1为例）：

benchmark模型推理工具，其输入是om模型以及模型所需要的输入bin文件，其输出是模型根据相应输入产生的输出文件。推理得到的结果会在test/result中。

将推理得到的结果重新转换为tensor形式，与标签mel_tgt计算mel_loss1。同时，将原模型pth权重前传得到的输出mel_out与标签mel_tgt计算出mel_loss2。mel_loss1与mel_loss2精度对齐则推理正确。

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd test
./benchmark.x86_64 -model_type=nlp -device_id=0 -batch_size=1 -om_path=./models/FastPitch_bs1.om -input_text_path=./input_bin_info.info -output_binary=True -useDvpp=False
```



### 5.3 精度对比

```
cd ..
python infer_test.py
```

以下为测试出的batch_size=1和16的精度对比：

```
mel_loss:
          om          pth
bs1	    11.246       11.265
bs16	11.330       11.265
```



## 6 性能对比

### 6.1 npu性能数据

1. 运行test/performance.sh脚本

```
cd test
./benchmark.x86_64 -round=20 -device_id=0 -batch_size=1 -om_path=./models/FastPitch_bs1.om
```

测试出来的ave_throughputRate，将其乘以4即为吞吐率。

以下计算结果为batch_size=1的结果。

![img](file:///C:\Users\1\AppData\Local\Temp\ksohtml\wps9EEB.tmp.jpg)







### 6.2 T4性能数据

提供以下测试代码作参考：

```python
import time

model=...(导入模型及加载pth权重)

input = torch.ones(size=(1, 200), dtype=torch.int64, device=device)
total_time = 0
lens = 20
for _ in range(lens):
	start = time.time()
	output = model(input)
	end = time.time()
	total_time += end - start
print(f"batch_size=1, FPS:{1.0/(total_time/lens)}")

 
input = torch.ones(size=(16, 200), dtype=torch.int64, device=device)
total_time = 0
lens = 20
for _ in range(lens):
	start = time.time()
	output = model(input)
	end = time.time()
	total_time += end - start
print(f"batch_size=16, FPS:{16.0/(total_time/lens)}")
```





### 6.3 性能对比

| Model     | Batch Size | A300 Throughput/Card | T4 Throughput/Card | A300/T4 |
| --------- | ---------- | -------------------- | ------------------ | ------- |
| FasfPitch | 1          | 54.1476              | 28.828             | 1.878   |
| FasfPitch | 4          | 51.728               | -                  | -       |
| FasfPitch | 8          | 51.3684              | -                  | -       |
| FasfPitch | 16         | 51.714               | 64.94              | 0.796   |
| FasfPitch | 32         | 52.0696              | -                  | -       |

由于模型并没有性能要求，bs1、bs4、bs8、bs16、bs32时npu的性能高于T4性能的0.5倍，性能达标。

