# 运行方法：

1.数据集准备

将数据集 LJSpeech-1.1解压并置于模型脚本根目录下，然后在模型脚本根目录下运行scripts/prepare_mels.sh

```
bash scripts/prepare_mels.sh
```

初次预处理时间较长，请耐心等待。


2.依赖

```
    LLVM
	DLLoger
	librosa
	numba
	llvmlite
```

(LLVM版本与numbra、llvmlite版本号严格依赖，如LLVM 7.0对应llvmlite的0.30.0，numbra的0.46.0版本)

3.启动训练

单p

```
bash run_1p.sh --train_epochs=训练周期数
```

训练日志被重定向为npu1p.log

8p

```
bash run_8p.sh --train_epochs=训练周期数
```

训练日志被重定向为npu8p.log