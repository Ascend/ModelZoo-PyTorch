# ADNet训练

```
## Atention-guided CNN for image denoising（ADNet）by Chunwei Tian, Yong Xu, Zuoyong Li, Wangmeng Zuo, Lunke Fei and Hong Liu is publised by Neural Networks (IF:8.05), 2020 (https://www.sciencedirect.com/science/article/pii/S0893608019304241) and it is implemented by Pytorch.


## This paper is pushed on home page of the Nueral Networks. Also, it is reported by wechat public accounts at  https://mp.weixin.qq.com/s/Debh7PZSFTBtOVxpFh9yfQ and https://wx.zsxq.com/mweb/views/topicdetail/topicdetail.html?topic_id=548112815452544&group_id=142181451122&user_id=28514284588581&from=timeline.

## This paper is the first paper via deep network properties for addressing image denoising with complex background.

## Absract
#### Deep convolutional neural networks (CNNs) have attracted considerable interest in low-level computer vision. Researches are usually devoted to improving the performance via very deep CNNs. However, as the depth increases, influences of the shallow layers on deep layers are weakened. Inspired by the fact, we propose an attention-guided denoising convolutional neural network (ADNet), mainly including a sparse block (SB), a feature enhancement block (FEB), an attention block (AB) and a reconstruction block (RB) for image denoising. Specifically, the SB makes a tradeoff between performance and efficiency by using dilated and common convolutions to remove the noise. The FEB integrates global and local features information via a long path to enhance the expressive ability of the denoising model. The AB is used to finely extract the noise information hidden in the complex background, which is very effective for complex noisy images, especially real noisy images and bind denoising. Also, the FEB is integrated with the AB to improve the efficiency and reduce the complexity for training a denoising model. Finally, a RB aims to construct the clean image through the obtained noise mapping and the given noisy image. Additionally, comprehensive experiments show that the proposed ADNet performs very well in three tasks (i.e., synthetic and real noisy images, and blind denoising) in terms of both quantitative and qualitative evaluations. The code of ADNet is accessible at https://github.com/hellloxiaotian/ADNet.
```

For more detail：https://www.sciencedirect.com/science/article/abs/pii/S0893608019304241



## Requirements

use pytorch, you can use pip or conda to install the requirements

```
# for pip
cd $project
pip3.7 install -r requirements.txt
CANN 5.0.3.alpha001
torchvision==0.6.0
```



## 数据集准备

1.从以下网址获取pristine_images_gray.tar.gz作为训练集

https://pan.baidu.com/s/1nkY-b5_mdzliL7Y7N9JQRQ

2.从以下网址获取BSD68作为标签

暂时没有在网上找到免费资源，可以从以下网址付费下载。

https://download.csdn.net/download/iteapoy/10902860

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
source ./test/env.sh
```

执行数据预处理脚本，将训练集图片裁剪成50*50的图片用与训练，运行成功会生成train.h5和val.h5文件，预处理需要h5py环境，请自行安装。

```
python3.7.5 preprocess.py --preprocess True --mode S
```



## TRAIN

### 单p训练

source 环境变量

```
source ./test/env.sh
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
source ./test/env.sh
```

性能脚本：

```
bash ./test/train_performance_8p.sh
```

精度脚本：

```
bash ./test/train_full_8p.sh
```

模型保存在

运行日志保存至./logssigma25.0_2021-09-05-19-23-13目录下（2021-09-05-19-23-13是运行train.py的时间，会根据当前时间自动更新），其中best_model.pth是在验证集上精度最高的模型。

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

请指定需要测试的模型路径，将--logdir参数设置为需要测试的模型目录。

## Demo
将一张图片放在./data/demo_img中，将--demo_pth_path设置为训练好的pth文件目录，执行以下程序。模型的运行结果保存在./data/demo_img/result文件夹里。
```
python3.7.5  demo.py --DeviceID 0  --num_of_layers 17 --test_noiseL 25 --demo_pth_path logssigma25.0_2021-09-03-10-39-34 
```

### 精度对比

由于NPU上使用torch.optim.Adam出现loss极大的情况，在使用apex.optimizers.NpuFusedSGD优化器后，loss正常，但是精度会有所损失。

|        | opt_level | loss_scale | optimizer                                                | PSNR  |
| ------ | --------- | ---------- | -------------------------------------------------------- | ----- |
| GPU-8p | o2        | 128        | optim.Adam                                               | 28.98 |
| GPU-8p | o2        | 128        | optim.SGD                                                | 27.83 |
| NPU-8p | 02        | 64         | apex.optimizers.NpuFusedAdam（不稳定，不能稳定复现精度） | 28.92 |
| NPU-8p | o2        | 8          | apex.optimizers.NpuFusedSGD                              | 28.49 |

在NPU上使用apex.optimizers.NpuFusedAdam不能稳定复现精度，有时候会出现loss极大的情况，导致训练失败。