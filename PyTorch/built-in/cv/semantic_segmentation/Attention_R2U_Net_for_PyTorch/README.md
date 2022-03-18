### pytorch Implementation of Attention R2U-Net

**Attention R2U-Net : Just integration of two recent advanced works (R2U-Net + Attention U-Net)**

## Attention R2U-Net
![AttR2U-Net](/img/AttR2U-Net.png)

## Code Reference

https://github.com/LeeJunHyun/Image_Segmentation

## Running
训练和验证步骤
### Step 1: Download DataSet

数据集下载路径 [ISIC 2018 dataset](https://challenge2018.isic-archive.com/task1/training/). 注意，仅仅需要下载2018年的Training Data和Training Ground Truth。下载完成并解压，修改路径参数后运行dataset.py将数据集划分为三部分，分别用于training, validation, and test, 三部分的比例是70%, 10% and 20%。数据集总共包含2594张图片， 其中1815用于training, 259 用于validation，剩下的520用于testing.  

### Step 2: Traing && Validation

脚本提供了单卡/8卡的训练脚本和test脚本。修改数据集路径参数后，运行train_1p.sh 和 train_8p.sh即可。其中数据集路径设置到train和test的父目录即可。训练完成后后，会在当前目录下生成训练日志和结果，单卡和8卡分别对应result_1p,result_8p. 文件下保存着最终训练完成的模型pkl文件。
修改test.sh的配置参数可以完成test过程。



### Step 3: Training Results


| DEVICE | FPS  | Npu_nums | Epochs | BatchSize | AMP  |  ACC  |
| :----: | :--: | :------: | :----: | :-------: | :--: | :---: |
|  V100  |  38  |    1     |  100   |    16     |  O2  |  NA   |
|  V100  | 304  |    8     |  100   |   16*8    |  O2  | 0.908 |
| NPU910 |  47  |    1     |  100   |    16     |  O2  |  NA   |
| NPU910 | 376  |    8     |  100   |   16*8    |  O2  |       |



注：

权重文件格式转换onnx，修改pthtar2onx.py文件中的第54行，将对应的值改为训练产生的pkl文件名
