# XCIT模型PyTorch离线推理指导
# 环境准备：

1.数据集

测试数据集为ImageNet的官方 2012的val数据集

下载evaluate所需的对应label文件(默认为target.json)，并放置在创建的推理结果保存文件夹中(默认为./result)。在后处理中会使用到对应文件。
```
mkdir result
mv target.json result
```
2.安装必要的依赖，模型部分依赖可通过requirement.txt安装，也可自行安装：
```
pip3.7 install -r requirements.txt
```
且模型要求torch版本较高:
```
pytorch>=1.8.0
onnx>=1.8.1
```
若要进行onnx端的性能测试，需要安装额外的第三方库:
```
pip install onnxmltools
```
3.获取模型代码
```
git clone https://github.com/facebookresearch/xcit.git
cd xcit
git checkout 82f5291f412604970c39a912586e008ec009cdca
patch -p1 < ../xcit.patch
cd ..
```

4.获取权重文件
在代码仓对应仓库下载指定权重文件（默认为xcit_p12_small),默认放置在pretrained文件夹中：
```
mkdir pretrained
mv {model}.pth pretrained 
```

5.获取benchmark工具
将benchmark.x86_64 benchmark.aarch64放在当前目录

6.创建保存结果文件夹
# 2 离线推理
310上执行，执行时确保device空闲
```
bash test/pth2om.sh
bash test/eval_acc_perf.sh --datasets_path=/root/datasets
```
评测结果：

|       模型        |  开源仓pth精度   |        310精度         |  性能基准   |   310性能   |
| :---------------: | :--------: | :--------------------: | :---------: | :---------: |
| XCIT bs1  | top1:82.0% | top1:81.9% top5:95.7% | 37.2 fps  | 121.3 fps |
| XCIT bs16 | top1:82.0% | top1:81.9% top5:95.7% | 293.0 fps | 200.9 fps  |
