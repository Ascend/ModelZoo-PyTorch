# Installation

We provide installation instructions for ImageNet classification experiments here.

## Dependency Setup
Create an new conda virtual environment
```
conda create --name convnext python=3.7.5 -y
conda activate convnext
```

Clone this repo and install required packages:
```
git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
```

The results in the paper are produced with `torch==1.8.1+ascend.rc2.20220505;torchvision==0.9.1;torch-npu 1.8.1rc2.post20220505 `.

## Dataset Preparation

Download the [ImageNet-1K](http://image-net.org/) classification dataset and structure the data as follows:
```
/path/to/imagenet-1k/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

For pre-training on [ImageNet-22K](http://image-net.org/), download the dataset and structure the data as follows:
```
/path/to/imagenet-22k/
  class1/
    img1.jpeg
  class2/
    img2.jpeg
  class3/
    img3.jpeg
  class4/
    img4.jpeg
```