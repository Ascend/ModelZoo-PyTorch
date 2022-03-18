# DCGAN模型PyTorch离线推理指导
## 1 环境准备
### 1.1 安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt    
```
### 1.2 获取，安装开源模型代码
```
git clone https://github.com/eriklindernoren/PyTorch-GAN.git  
```
使用patch文件更改开源代码仓源码
```
mv dcgan.patch PyTorch-GAN/
cd PyTorch-GAN/
git apply dcgan.patch
cd ..
```
### 1.3 获取权重文件  
[DCGAN预训练权重文件](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/GAN/DCGan/checkpoint-amp-epoch_200.pth)
```
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/GAN/DCGan/checkpoint-amp-epoch_200.pth
```
### 1.4 数据集  
DCGAN的输入是随机噪声。当前目录下的`dcgan_preprocess.py`文件会随机生成输入噪声作为数据集。
此脚本无需主动运行。

默认设置下，生成8192个噪声样本。
### 1.5 [获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  
将benchmark.x86_64或benchmark.aarch64放到当前目录，并更改权限
```
chmod 777 benchmark.x86_64
```
## 2 离线推理
### 2.1 性能测试
310上执行，执行时使npu-smi info查看设备状态，确保device空闲
```
bash test/pth2om.sh
bash test/eval_perf.sh
```
### 2.2 精度测试
由于开源代码仓并未提供合适的精度指标来衡量模型的生成精度。
我们提供了图像像素差均值(mean)和图像余弦相似度(consine)作为精度指标以供参考。

因为npu的推理结果是以pth的生成结果为基准。
所以两个指标的计算对象分别是pth模型在cpu上的生成集合和om模型在npu上的生成集合。
除却均值指标与相似度指标外，还提供了一个精度指标(acc)。`acc=(cosine+1)/2`。目的是为了获得一个百分比值便于直观理解精度。
```
#直接执行acc验证脚本
bash test/eval_acc.sh
```

结果分别保存在当前目录的`dcgan_acc_eval_bs1.log`与`dcgan_acc_eval_bs16.log`中。
### 2.3 测评结果
|模型|精度(mean)|精度(cosine)|精度(acc)|性能基准|310性能|
|----|----|----|----|----|----|
|DCGAN bs1|0.0004|1.0|100.0%|10174.65fps|11429.32fps|
|DCGAN bs16|0.0004|1.0|100.0%|46711.51fps|63607.60fps|

