CGAN训练

```
Generative adversarial nets can be extended to a conditional model if both the generator and discriminator are conditioned on some extra information y. y could be any kind of auxiliary information,such as class labels or data from other modalities. The author perform the conditioning by feeding y into the both the discriminator and generator as additional input layer.In the generator the prior input noise pz(z), and y are combined in joint hidden representation, and the adversarial training framework allows for considerable flexibility in how this hidden representation is composed. In the discriminator x and y are presented as inputs and to a discriminative function.
```

For more detail：https://arxiv.org/abs/1411.1784

The original gpu code:https://github.com/znxlwm/pytorch-generative-model-collections/

## Requirements

use pytorch, you can use pip or conda to install the requirements

```
# for pip
cd $project
pip3.7 install -r requirements.txt
CANN 5.0.3
torchvision
```
注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision ,建议Pillow版本是9.1.0 torchvision版本是0.6.0


## 数据集准备

1.下载mnist数据集作为训练集，dataloader.py中有自动下载mnist数据集的代码，执行训练命令会自动调用dataloader.py下载数据集，并保存在“./data/mnist“目录下。（请保持网络通畅）

文件结构如下：


```
CGAN
|-- data                               /数据集文件夹
|   |-- mnist                           /验证集，测试集
|-- demo                               /demo.py的输出
|--models                              /生成器和判别器模型保存目录
|-- test                               /脚本文件夹
|   |--env.sh                          /环境配置文件
|   |--eval_1p.sh                      /单卡测试脚本
|   |--train_full_1p.sh                /单卡精度测试脚本
|   |--train_full_8p.sh                /8卡精度测试脚本
|   |--train_performance_1p.sh         /单卡性能测试脚本
|   |--train_performance_8p.sh         /8卡性能测试脚本
|--results                             /生成器生成图片保存路径
|-- CGAN.py                            /模型定义脚本
|-- demo.py                            /例子脚本
|-- dataloaderpy                       /数据预处理文件
|-- main.py                            /主函数，训练启动脚本
|-- utils.py                           /其它需要调用的函数脚本
```



## TRAIN

### 单p训练

source 环境变量

```
source ./test/env_npu.sh
```

性能脚本：

```
bash ./test/train_performance_1p.sh
```

精度脚本：

```
bash ./test/train_full_1p.sh
```



### 多p训练

source 环境变量

```
source ./test/env_npu.sh
```

性能脚本：

```
bash ./test/train_performance_8p.sh
```

精度脚本：

```
bash ./test/train_full_8p.sh
```

模型保存在”./models“目录下，模型生成的图片保存在”./result“目录下

模型训练的loss曲线保存在”./models"目录下。

## TEST

对比GPU和NPU模型生成的图片和训练loss曲线，两者大致一致。

| name        | Epoch 50                                            | GIF                                                          | Loss                                            |
| :---------- | --------------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------- |
| CGAN on GPU | ![](README.assets/CGAN_epoch050-16371345386081.png) | ![](README.assets/CGAN_generate_animation-16371345738152.gif) | ![](README.assets/CGAN_loss-16371346002224.png) |
| CGAN on NPU | ![](README.assets/CGAN_epoch050-16371346136555.png) | ![](README.assets/CGAN_generate_animation-16371346226546.gif) | ![](README.assets/CGAN_loss-16371346305157.png) |

## Pth2onnx

执行以下命令，完成pth到onnx模型的转换

```
python3 pth2onnx.py --pth_path ./models/mnist/CGAN/CGAN_G.pth --onnx_path ./CGAN.onnx
```

## Demo

执行以下命令，程序会自动生成输入并经过网络产生输出，将输出保存在"demo/demo_result.png"中
```
python3 demo.py --pth_path ./models/mnist/CGAN/CGAN_G.pth --save_path ./demo
```

### 精度对比

对比GPU和NPU生成的图片和loss曲线，两者差异不大，精度达标。

