# EfficientNet-B1模型PyTorch离线推理指导
# 环境准备：

1.数据集

测试数据集为ImageNet的官方 2012的val数据集，5w张图片放置在一个文件夹下，并由官方对应的 ILSVRC2012_devkit_t12 文件夹。

第一个参数为 新下载且未分类的 imagenet的val 数据集路径，

第二个参数为官方 提供的 devkit 文件夹，如果要保留val文件夹请先备份

```
python3.7 ImageNet_val_split.py ./val ./ILSVRC2012_devkit_t12
```

2.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装
```
pip3.7 install -r requirements.txt
```
3.获取模型代码
```
git clone https://github.com/facebookresearch/pycls
cd pycls
git reset f20820e01eef7b9a47b77f13464e3e77c44d5e1f --hard
cd ..
```

4.获取权重文件
```
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/image_classification/EfficientNet-B1/EN-B1_dds_8gpu.pyth
```
5.获取benchmark工具
将benchmark.x86_64 benchmark.aarch64放在当前目录

# 2 离线推理

310上执行，执行时确保device空闲
```
bash test/pth2om.sh
bash test/eval_acc_perf.sh --datasets_path=/root/datasets
```
评测结果：

|       模型        |  开源仓pth精度   |        310精度         |  性能基准   |   310性能   |
| :---------------: | :--------: | :--------------------: | :---------: | :---------: |
| Efficient-B1 bs1  | top1:75.9% | top1:75.5% top5:92.78% | 694.137fps  | 940.524fps |
| Efficient-B1 bs16 | top1:75.9% | top1:75.5% top5:92.78% | 1408.138fps | 1490.54fps  |

