# Speech-Transformer模型PyTorch离线推理指导

## 1 环境准备

1. 安装kaldi

- `git clone https://github.com/kaldi-asr/kaldi`

- 请根据INSTALL文件中Option 1安装

2. 获取并修改开源模型代码  

```shell
git clone https://github.com/kaituoxu/Speech-Transformer
```
- **将所有文件放在开源仓Speech-Transformer/egs/aishell**下面

```shell
cd Speech-Transformer
patch -p1 < egs/aishell/SpeechTransformer.patch
```
3. 安装依赖

```shell
cd Speech-Transformer/egs/aishell
pip install requirements.txt
```

4. 获取权重文件pth权重文件

```shell
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/audio/speech-transformer/final.pth.tar
```

- 将权重文件放在Speech-Transformer/egs/aishell下

5. 获取aishell数据集

- [aishell](http://www.openslr.org/33/) 下载data_aishell.tgz 解压，然后再解压data_aishell/wav 目录下面所有的文件

6. 提取特征

```shell
cd Speech-Transformer/tools
make KALDI=/path/to/kaldi #指向kaldi的源码
cd ../egs/aishell
# 修改run.sh 中data变量指向aishell数据集，例如，data_aishell放在/home目录下面，则修改data=/home
bash run.sh
```

## 2 离线推理

310上执行，执行时使npu-smi info查看设备状态，确保device空闲

```shell
# 转成onnx
bash test/pth2onnx.sh
# 转成om
bash test/onnx2om.sh
# 28-36 行得到精度数据
# 38-41 行得到性能数据
bash test/eval_acc_perf.sh
```


评测结果：
| 模型 | 在线推理pth精度 | 310离线推理精度  | 基准性能  | 310性能  |
|----|---------|---|---|---|
|  Speech-Transformer  |    9.8     | 9.9  | 0.83fps | 0.82fps |

备注：
1. 模型不支持多batch
2. 精度测评脚本包含了精度和性能结果, 结果中Err即为精度
3. 基准性能获取方法

```shell
pip install onnxruntime-gpu
bash test/eval_acc_perf_onnx.sh
```

