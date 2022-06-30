
# ADNet训练

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Image denoising**

**版本（Version）：1.1**

**修改时间（Modified） ：202205.28**

**大小（Size）：1M**

**框架（Framework）：Pytorch 1.8.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于Pytorch框架的ADNet图像去噪网络训练代码** 


For more detail：https://www.sciencedirect.com/science/article/abs/pii/S0893608019304241


## Requirements

use pytorch, you can use pip or conda to install the requirements

```
# for pip
cd $project
pip3.7 install -r requirements.txt
CANN5.1.RC1 
torchvision==0.6.0
```



## 数据集准备

请用户自行准备好数据集，包含训练集和验证集两部分，pristine_images_gray作为训练集，BSD68作为标签验证集。

文件结构如下：

```
ADNET
|-- data
|   |-- BSD68
|   |-- pristine_images_gray   
|   |-- demo_img 
|       |--result
|-- test      
|-- dataset.py
|-- demo.py
|-- models.py
|-- preprocess.py
|-- test.py
|-- train.py
|-- utils.py

```

将数据集按照以上结构放在代码目录下

## 处理数据

source环境变量

```
source ./test/env_npu.sh
```

执行数据预处理脚本，将训练集图片裁剪成50*50的图片用与训练，运行成功会生成train.h5和val.h5文件。

```
python3.7 preprocess.py --preprocess True --mode S
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

模型保存在./logssigma25.0_2022-05-28-18-44-03目录下（2022-05-28-18-44-03是运行train.py的时间，会根据当前时间自动更新），其中best_model.pth是在验证集上精度最高的模型。

## TEST

测试精度 

使用sh文件

```
bash test/eval_1p.sh
```

测试之前请指定测试的模型路径。打开./test/eval.sh文件，如下所示：

```
python3.7.5  test.py --is_distributed 0 --DeviceID 0 --num_gpus 1 --num_of_layers 17 --logdir logssigma25.0_2021-08-31-19-13-09 --test_data BSD68 --test_noiseL 25 | tee -a eval_1p.log
```

请指定需要测试的模型路径，将--logdir参数设置为需要测试的模型目录，即如上文所列举的设置--logdir ./logssigma25.0_2022-05-28-18-44-03。

## Demo
将一张图片放在./data/demo_img中，将--demo_pth_path设置为训练好的pth文件目录，执行以下程序。模型的运行结果保存在./data/demo_img/result文件夹里。
```
python3.7  demo.py --DeviceID 0  --num_of_layers 17 --test_noiseL 25 --demo_pth_path logssigma25.0_2021-09-03-10-39-34 
```

### 精度对比

由于NPU上使用torch.optim.Adam出现loss极大的情况，在使用apex.optimizers.NpuFusedSGD优化器后，loss正常，但是精度会有所损失。

|        | opt_level | loss_scale | optimizer                                                | PSNR  |
| ------ | --------- | ---------- | -------------------------------------------------------- | ----- |
| NPU-8p | o2        | 4          | apex.optimizers.NpuFusedSGD                              | 28.52 |
| NPU-8p | o2        | dynamic    | apex.optimizers.NpuFusedSGD                              | 28.45 |

### 性能对比

由于NPU上使用torch.optim.Adam出现loss极大的情况，在使用apex.optimizers.NpuFusedSGD优化器后，loss正常，但是精度会有所损失。

|        | opt_level | loss_scale | torch version                                            | FPS   |
| ------ | --------- | ---------- | -------------------------------------------------------- | ----- |
| NPU-1p | o2        | 8          | 1.8.1                                                | 1640.7331 |
| NPU-1p | o2        | 8          | 1.5.0                                                | 1305.5087 |
| NPU-1p | o2        | dynamic    | 1.8.1                                                | 1507.3948 |
| NPU-1p | o2        | dynamic    | 1.5.0                                                | 1749.3947 |

