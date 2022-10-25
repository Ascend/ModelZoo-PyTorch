# FSRCNN训练

## 概述

### 简述

​	超分辨率超卷积神经网络(SRCNN)作为一种成功应用的图像超分辨率(SR)深度模型，在速度和恢复质量上表现出了比之前更好的性能 。然而，高的计算成本仍然阻碍了它在需要实时性能的场景下使用。FSRCNN是一种紧凑的沙漏形CNN结构，可以实现更快、更好的进行SR。该模型提高了40倍以上的速度，恢复质量更好

- 参考论文：[Accelerating the Super-Resolution Convolutional Neural Network (arxiv.org)](https://arxiv.org/abs/1608.00367)
- 参考链接：[yjn870/FSRCNN-pytorch: PyTorch implementation of Accelerating the Super-Resolution Convolutional Neural Network (ECCV 2016) (github.com)](https://github.com/yjn870/FSRCNN-pytorch)

### 默认配置

- 训练超参：
  - batch_size: 16
  - lr: 1e-3
  - num-epochs: 20
  - num-workers: 8
  - scale:  3 （根据需要设置为2,3,4。该参数进行修改时需要同时修改使用的训练集和验证集）

## 训练环境准备

- 硬件环境和运行环境准备请参见《[CANN软件安装指南](https://gitee.com/link?target=https%3A%2F%2Fsupport.huawei.com%2Fenterprise%2Fzh%2Fascend-computing%2Fcann-pid-251168373%3Fcategory%3Dinstallation-update)》
- 需要安装以下依赖
  - Numpy 1.15.4
  - Pillow 5.4.1
  - h5py 2.8.0
  - tqdm 4.30.0

## 快速上手

### 数据集准备

使用的训练集为91-image，验证集为Set5。

下载完数据集后运行prepare.py文件转换为h5py格式的数据。

```
python3 prepare.py --images-dir "images-dir" \
                --output-path "data/Set5_x3.h5" \
                --scale 3 
```

或者直接下载h5py格式的数据集，转换为h5py的数据下载链接为：URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=RrSfPzFZtvwfARxmYDFuJNIKT5r8/eVrwI26J888X/6jNdT/aKnEcd1TmxUXcXRcMIZsURCs6eof5DSB8E7Rr79//dbTrzGBA8ezuqVu/FUYvM6rlAgcEUeUWVTgUFJPpg+g9AutmQi+arIZTT2Qr7nokx1jQIuy5nG3HNOhcQCLlxuARBfiMBsSX8xqqlZAjrwhG9THZbIVYWYnzOJiJ25WN24xZrESi5ohzDKjnoys/tqFtbnB1quBd77o15lu4J1kR46XYc2YOgiJFYy+gGUA76fnCu8Fj2Z6B46jR/zgSyWdus33m1oTEwAt6m+f8i4kiY29mN04aqOZiWOplKctvzrZhxqW/K4kT5l9P0t/a7xoJzyvxw/OxHGcZR1nRW1KHvjTyhcbQD3iqaYq1n85Qy30J2j7ct+DXjmWOhuyd3u4oNVGXutEHfdtb18nVLXK4J0xt6A0UQAo/NyHIvTnEcZJ8K535ZV4FDh6WIiUfSWyBF7lQNAbzRyjaNnCOiBosIB8g/0T1g1bxq3Votb3N4UphRSuYedngVeoLZQ=

提取码:
123456

*有效期至: 2023/08/25 09:41:21 GMT+08:00

### 模型训练

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 开始训练。

  - 启动训练之前，首先要配置程序运行相关环境变量。

    环境变量配置信息参见：

    [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend 910训练平台环境变量设置?sort_id=3148819)

  - 单卡训练

    1、 根据数据集实际路径配置data_path，使用命令行进行训练：

    ​	训练时scale的值应该与train-file和eval-file选用的数据集保持一致。例如scale=4，则train-file为91-image_x4.h5，eval-file为Set5_x4.h5。

    ```bash
    python3 train.py --train-file "data/91-image_x3.h5" \
                    --eval-file "data/Set5_x3.h5" \
                    --outputs-dir "/outputs" \
                    --scale 3 \
                    --lr 1e-3 \
                    --batch-size 16 \
                    --num-epochs 20 \
                    --num-workers 8 \
                    --seed 123 
    ```

    2、精度指标

    | PSNR   | 论文  | GPU  | NPU   |
    | ------ | ----- | -------- | ----- |
    | Scale2 | 37.12 | 37.06    | 37.03 |
    | Scale3 | 33.22 | 33.61    | 33.56 |
    | Scale4 | 30.50 | 30.48    | 30.45 |

    3、性能指标

    | GPU   | NPU       |
    | --------- | --------- |
    | 2500 it/s | 3800 it/s |

    

## 高级参考

### 脚本和示例代码

```bash
│-datasets.py
│-LICENSE
│-models.py  模型定义
│-prepare.py h5py文件准备
│-README.md 
│-requirements.txt 所需包
│-test.py  测试单张图片效果
│-train.py 模型训练函数
│-utils.py 工具类
│-data 数据存储文件夹
├── test
│    ├──train_full_1p.sh                      //单卡全量训练启动脚本
```

​		



