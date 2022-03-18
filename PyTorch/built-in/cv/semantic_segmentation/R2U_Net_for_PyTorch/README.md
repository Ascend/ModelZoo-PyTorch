###  pytorch Implementation of R2U-Net

** ** R2U-Net: Learning Where to Look for the Pancreas** **

## R2U-Net

## Code Reference

https://github.com/LeeJunHyun/Image_Segmentation

## Running

训练和验证步骤

### Step 1: Download DataSet

数据集下载路径 [ISIC 2018 dataset](https://challenge2018.isic-archive.com/task1/training/). 注意，仅仅需要下载2018年的Training Data和Training Ground Truth。下载完成并解压，修改路径参数后运行dataset.py将数据集划分为三部分，分别用于training, validation, and test, 三部分的比例是70%, 10% and 20%。数据集总共包含2594张图片， 其中1815用于training, 259 用于validation，剩下的520用于testing.

### Step 2: Traing && Validation

脚本提供了单卡/8卡的训练脚本和test脚本。修改数据集路径参数后，运行train_1p.sh 和 train_8p.sh即可。其中数据集路径设置到train和test的父目录即可。训练完成后后，会在当前目录下生成训练日志和结果，单卡和8卡分别对应result_1p,result_8p. 文件下保存着最终训练完成的模型pkl文件。 修改test.sh的配置参数可以完成test过程。

### Step 3: Training process

单卡训练流程：

    1.安装环境
    2.修改参数device_id（单卡训练所使用的device id），为训练配置device_id，比如device_id=0
    3.开始训练
        bash ./test/train_full_1p.sh  --data_path=数据集路径  --epoch=训练次数      # 精度训练
        bash ./test/train_performance_1p.sh  --data_path=数据集路径 --epoch=训练次数 # 性能训练


多卡训练流程

    1.安装环境
    2.开始训练
        bash ./test/train_full_8p.sh  --data_path=数据集路径  --epochs=训练次数       # 精度训练
        bash ./test/train_performance_8p.sh  --data_path=数据集路径 --epochs=训练次数 # 性能训练

### Step 4: Training Results

训练日志路径：网络脚本test下output文件夹内。例如：
      test/output/devie_id/train_${device_id}.log          # 训练脚本原生日志
      test/output/devie_id/R2U_Net_bs1024_8p_perf.log  # 8p性能训练结果日志
      test/output/devie_id/R2U_Net_bs1024_8p_acc.log   # 8p精度训练结果日志

训练模型：训练生成的模型默认会写入到和test文件同一目录下。当训练正常结束时，checkpoint.pth.tar为最终结果。

| DEVICE | FPS  | Npu_nums | Epochs | BatchSize | AMP  | ACC   |
| ------ | ---- | -------- | ------ | --------- | ---- | ----- |
| V100   | 35   | 1        | 150    | 16        | O2   | 0.88 |
| V100   | 304  | 8        | 150    | 16*8      | O2   | 0.88 |
| NPU910 | 45   | 1        | 150    | 16        | O2   | 0.90 |
| NPU910 | 369  | 8        | 150    | 16*8      | O2   | 0.89 |

