# DCGAN

该目录为DCGAN在MNIST数据集上的训练与测试，主要参考实现[eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)

##  DCGAN相关细节

* 添加DDP多卡训练代码
* 添加混合精度训练
* 修改了训练逻辑，以解决原代码在多卡训练时的因数据被改动而无法反向传播的问题
* 添加相关日志代码，模型保存代码

## 环境准备

* 执行本样例前，请确保已安装有昇腾AI处理器的硬件环境，CANN包版本5.0.3
* 该目录下的实现是基于PyTorch框架，其中torch版本为1.5.0+ascend.post3.20210930，使用的混合精度apex版本为0.1+ascend.20210930,
torchvision版本为0.2.2.post3。

## 训练准备-关于数据集

* 此DCGAN模型在MNIST的手写数据集上训练。
* 数据集来自pytorch中的数据集接口，若指定的目录下无数据集，程序将会自动下载数据集
* 参考一下命令下载数据集
```shell
bash test/get_mnist.sh --data_path=/path/to/data/root/
# 例:bash test/get_mnist.sh --data_path=./data/
```

## 快速运行

模型的训练文件详见main.py, 运行以下脚本能够进行单/多卡的训练和性能测试:

```shell
# data_path参数可任意指定一目录，建议为“./data/”
# train 1p performance
bash test/train_performance_1p.sh --data_path=/path/to/data/root/
# train 1p full
bash test/train_full_1p.sh --data_path=/path/to/data/root/
# 例：bash test/train_full_1p.sh --data_path=./data/

# train 8p performance
bash test/train_performance_8p.sh --data_path=/path/to/data/root/
# train 8p full
bash test/train_full_8p.sh --data_path=/path/to/data/root/

# 验证脚本使用保存的模型随机生成10张图片，
bash test/eval.sh --checkpoint_path=./checkpoint-amp-epoch_20.pth
```

## DCGAN的训练结果
### 精度结果-生成图片展示
更多结果请参看`./imgs`文件夹

![sample](./imgs/008.jpg)

### 性能结果
此处展示不同batchsize下，DCGAN模型在NPU下的FPS。提高batchsize的目的是在保证精度的情况下尽可能利用NPU性能。

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |Batchsize|
| :------: | :------:  | :------: | :------: | :------: |:------:|
| -        | 1008.340      | 1        | 20      | O2    |64|
| -        | 3099.795      | 1        | 20      | O2    |128|
| -     | 6258.619     | 8        | 20      | O2      |512|
| -     | 12742.566     | 8        | 20      | O2      |1024|

