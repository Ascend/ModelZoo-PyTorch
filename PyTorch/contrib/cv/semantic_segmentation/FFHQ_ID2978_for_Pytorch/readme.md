# FFHQ-Aging-Dataset
- Paper:[[2003.09764] Lifespan Age Transformation Synthesis](https://arxiv.org/abs/2003.09764)
- Github Code:[royorel/FFHQ-Aging-Dataset: FFHQ-Aging Dataset](https://github.com/royorel/FFHQ-Aging-Dataset)

Face Semantic maps were acquired by training a pytorch implementation of [DeepLabV3](https://github.com/chenxi116/DeepLabv3.pytorch) network on the [CelebAMASK-HQ](https://github.com/switchablenorms/CelebAMask-HQ) dataset.

## Directory structure
```
.
├── data_loader.py	# 数据集加载
├── deeplab_model	#存放模型参数，下载的模型请放这里。
│   ├── deeplab_model.pth
│   └── R-101-GN-WS.pth.tar
├── ffhq_aging128×128	#存放数据集，下载的数据集请解压到这里
├── deeplab.py	#deeplap v3模型脚本
├── readme.md	
├── requirements.txt
├── run_deeplab.py
└── utils.py
```
## Environment preparation
- Install Packages
  - pip install -r requirements.txt
- Download **FFHQ-Aging-Dataset** & **Deeplab Model** from [original repo](https://github.com/royorel/FFHQ-Aging-Dataset) & [deeplab_model/R-101-GN-WS.pth.tar](https://drive.google.com/uc?id=1oRGgrI4KNdefbWVpw0rRkEP1gbJIRokM) & [deeplab_model/deeplab_model.pth](https://drive.google.com/uc?id=1w2XjDywFr2NjuUWaLQDRktH7VwIfuNlYhttps://drive.google.com/uc?id=1w2XjDywFr2NjuUWaLQDRktH7VwIfuNlY)
  - The original **FFHQ-dataset** is stored on the [google drive](https://drive.google.com/drive/folders/1u2xu7bSrWxrbUxk-dT-UvEJq8IjdmNTP), By running the [original repo's get_ffhq_aging.sh](https://github.com/royorel/FFHQ-Aging-Dataset/blob/master/get_ffhq_aging.sh) , you can easily get **FFHQ-Aging-Dataset**.

## Run

python3 run_deeplab.py


> 分割后的图像，放在原图片存放路径里的parsings文件夹下。

> 例如：ffhq_aging128×128\0-2\parsings    
>    ffhq_aging128×128\3-6\parsings等

### Runing result

<!-- - 原始图像、GPU分割效果如下：
![图像分割](https://github.com/royorel/FFHQ-Aging-Dataset/blob/master/images/dataset_samples_github.png) -->

- 原始图像、GPU分割、NPU分割效果对比如下：
<br> 调用deeplab模型后，原始图片被分割为：背景、皮肤、鼻子、眼睛等部分。

# Statement
For details about the public address of the code in this repository, you can get from the file public_address_statement.md
