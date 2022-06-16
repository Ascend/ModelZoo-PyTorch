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
注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision
建议：Pillow版本是9.1.0 torchvision版本是0.6.0
其中数据集路径设置到train和test的父目录即可。 文件下保存着最终训练完成的模型pkl文件。 
开始训练：
    bash ./test/train_full_1p.sh  --data_path=数据集路径                    # 单卡训练
    bash ./test/train_full_8p.sh  --data_path=数据集路径                    # 8卡训练
### Step 3: Training Results


| DEVICE | FPS  | Npu_nums | Epochs | BatchSize | AMP  |  ACC  |
| :----: | :--: | :------: | :----: | :-------: | :--: | :---: |
|  V100  |  38  |    1     |  100   |    16     |  O2  |  NA   |
|  V100  | 304  |    8     |  100   |   16*8    |  O2  | 0.908 |
| NPU910 |  47  |    1     |  100   |    16     |  O2  |  NA   |
| NPU910 | 376  |    8     |  100   |   16*8    |  O2  |       |

### Step 4: Training Logs

/home/Attention_R2U_Net_for_PyTorch/test/output/

注：

权重文件格式转换onnx，修改pthtar2onx.py文件中的第54行，将对应的值改为训练产生的pkl文件名
