# LResNet100E-IR 训练指导



## 1 模型概述

### 1.1 论文地址

[LResNet100E-IR论文](https://arxiv.org/pdf/1801.07698.pdf )

### 1.2 代码地址

[LResNet100E-IR代码](https://github.com/TreB1eN/InsightFace_Pytorch )



## 2 环境说明

```
pytorch
torchvision
opencv-python==4.5.3.56
easydict==1.9
cython==0.29.24
packaging>=21.0
setuptools>=52.0.0
bcolz==1.2.1
tqdm==4.62.2
scikit-learn==0.24.2
tensorboardX==2.4
matplotlib==3.4.3
numpy==1.21.2
onnx==1.10.1
apex
```



## 3 训练准备

### 3.1 代码准备

```bash
# 切换到工作目录,假设工作目录为 LResNet100E-IR
cd LResNet100E-IR
# 创建日志存储，模型存储目录
rm -rf ./work_space/* 
mkdir ./work_space/history && mkdir ./work_space/log && mkdir ./work_space/models && mkdir ./work_space/save
```

### 3.2 获取数据

+ 参考源码仓的方式获取数据集 https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/README.md



### 3.3 主要脚本和代码说明

```
|———— test
|  |---- output							//训练验证时生成，保存训练和验证日志
|  |---- env_npu.sh						//环境变量配置文件
|  |---- train_eval_8p.sh				//8P验证脚本
|  |---- train_finetune_1p.sh			//1P迁移脚本
|  |---- train_full_8p.sh				//8P训练脚本
|  |---- train_performance_1p.sh		//1P测试脚本
|  |---- train_performance_8p.sh		//8P测试脚本
|———— data
|  |---- data_pipe.py					//数据集加载代码
|  |---- faces_emore					//数据集
|———— work_space				
|  |---- log							//默认tensorboard日志保存路径
|  |---- models							//默认模型保存路径					
|———— config.py							//配置文件
|———— demo.py							//测试代码
|———— model.py							//网络模型代码
|———— prepare_data.py					//数据集处理代码
|———— eval.py							//验证代码入口
|———— LResNet_pth2onnx.py				//模型转换代码
|———— train.py							//训练代码入口
|———— prof.py							//输出profiling代码
|———— Learner.py						//训练代码
|———— utils.py							//其余辅助类代码
|———— verifacation.py					//验证代码
```



## 4 训练及验证

环境变量配置

```bash
source test/env_npu.sh
```

1p测试

```bash
bash test/train_performance_1p.sh --data_path=data/faces_emore
```

8p测试

```bash
bash test/train_performance_8p.sh --data_path=data/faces_emore
```

1p训练，大约花费40h

```bash
bash test/train_full_1p.sh --data_path=data/faces_emore
```

8p训练，大约花费7h

```bash
bash test/train_full_8p.sh --data_path=data/faces_emore
```

8p验证，请自行添加对应的 pth_path地址

```bash
bash test/train_eval_8p.sh --data_path=data/faces_emore/lfw.bin --pth_path=
```

1p迁移学习，请自行添加对应的 pth_path地址

```bash
bash test/train_finetune_1p.sh --data_path=data/faces_emore --pth_path=
```



## 5 推理

1.简单的测试下模型输出是否保持一致, 输入图片，输出tensor

根据实际情况修改 weights_path

```bash
python demo.py --weights_path=model_ir_se100.pth --data_path=demo_img.jpg
```

2.测试下输入图片的人脸是否在数据库

- 准备数据库：如下所示，其中图片size为112x112

```
|-facebank
|  |- name1
|  |  |- name1.jpg
|  |- name2
|  |  |- name3.jpg
|  |- name3
|  |  |- name3.jpg
```

* weights_path 参数：模型权重路径

* data_path 参数：测试图片路径，注意图片 size 需为112x112

* check 参数： 是否进行与人脸数据库对比

* update 参数：更新人脸数据库（加入人脸，更换模型，第一次使用，都需要设置为1）

* facebank_dir 参数：人脸数据库目录

* threshold 参数：阈值（由验证得来）

```bash
python demo.py --weights_path=model_ir_se100.pth \
	--data_path=demo_img.jpg \
	--check=1 \
	--update=1 \
	--facebank_dir=facebank \
	--threshold 1.54
```



## 6 pth转onnx

pth权重转onnx权重，参数1为pth模型权重地址，参数2为输入onnx模型权重地址，参数3为batch size

命令示例：

参数1：pth模型权重路径

参数2：转出的onnx模型权重路径

参数3：batch size大小

```bash
python LResNet_pth2onnx.py model_ir_se100.pth model_ir_se100.onnx 1
```



## 7 训练精度及性能

| NPU nums (Ascend 910) | Epoch | BatchSize | lr    | Accuracy | FPS    | amp mode |
| --------------------- | ----- | --------- | ----- | -------- | ------ | -------- |
| 1                     | 20    | 256       | 0.001 |          | 505.43 | O2       |
| 8                     | 20    | 320*8     | 0.01 | 0.9967   | 4696   | O2       |

