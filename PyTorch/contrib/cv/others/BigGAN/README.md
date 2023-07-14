# BigGAN

该目录为BigGAN在imagenet2012数据集上的训练与测试，主要参考实现[ajbrock/BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch)

##  BigGAN相关细节

* 添加DDP多卡训练代码
* 增加模型精度验证脚本

## 环境准备

* 执行本样例前，请确保已安装有昇腾AI处理器的硬件环境，CANN包版本5.0.3
* 该目录下的实现是基于PyTorch框架，其中torch版本为1.5.0+ascend.post3
* pip install -r requirements.txt
    注：pillow建议安装较新版本， 与之对应的torchvision版本如果无法直接安装，可使用源码安装对应的版本，源码参考链接：https://github.com/pytorch/vision ,建议Pillow版本是9.1.0 torchvision版本是0.6.0

## 训练准备

* 下载imagenet2012数据集
* 处理imagenet2012数据集，将数据集裁剪为128大小，并保存为h5文件。同时使用pytorch的inception模型接口计算每张图像的特征。
```shell
bash prepare_data.sh --data_path=/path/to/imagenet
```

注：/path/to/imagenet路径下应该有“ImageNet”的文件夹，文件夹下包含一个名为“train”的文件夹包含imagenet训练数据。

## 快速运行

模型的训练文件详见train.py, 运行以下脚本能够进行单/多卡的训练和性能测试:

（由于脚本自动删除已存在目录，执行完1p的性能脚本，记得保存日志文件）

以下脚本以h5数据集文件位于./data路径为例。
```shell
# train 1p performance,
bash test/train_performance_1p.sh --data_path=/path/to/imagenet

# train 8p full
bash test/train_full_8p.sh --data_path=/path/to/imagenet


# 验证某次迭代保存的权重的精度
# biggan的所有权重保存在"./weights/biggan_full"路径下，该路径下以迭代次数命名文件夹，例如验证1500次迭代的模型精度执行以下命令
# 本次的验证结果保存在主目录下“evaluation.log”文件中
bash test/eval.sh --weights_path=./weights/biggan_full/1500
```

## BigGAN的训练结果

训练时模式为FP32

8p-npu，1500次迭代模型的精度

|   IS   |   FID    |  FPS   | Npu_nums | iters | AMP_Type |
|:------:|:------:|:------:|:--------:|:-----:|:--------:|
|   -    |    -     | 0.492  |    1     |  10   |   FP32   |
|  1.51  | 278.3 | 3.673  |    8     | 1500  |   FP32   |

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md