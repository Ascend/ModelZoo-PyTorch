# Swin-Transformer 模型PyTorch离线推理指导
# 环境准备：

1.数据集

测试数据集为ImageNet的官方 2012的val数据集

下载evaluate所需的对应label文件(默认为target.json)，并放置在创建的推理结果保存文件夹中(默认为./result)。在后处理中会使用到对应文件。
```
mkdir result
mv target.json result
```
2.安装必要的依赖，模型部分依赖可通过requirement.txt安装，也可自行安装。此外，模型性能的优化需要使用到华为的onnx_tools工具：
```
pip3.7 install -r requirements.txt
git clone https://gitee.com/zhang-dongyu/onnx_tools.git
cd onnx_tools
git reset --hard 8765f572a221e3f63b4e36f51b7c5eee86435d1e
cd ..
```
且模型要求torch版本较高:
```
pytorch>=1.8.0
onnx>=1.10.1
```
3.获取模型代码
```
git clone https://github.com/microsoft/Swin-Transformer.git
cd Swin-Transformer
git reset --hard 6bbd83ca617db8480b2fb9b335c476ffaf5afb1a
patch -p1 < ../swin.patch
cd ..
```

4.获取权重文件
在代码仓对应仓库下载指定权重文件（默认为swin_tiny_patch4_window7_224.pth),默认放置在resume目录下：
```
mkdir resume
mv {model}.pth resume 
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
| Swin-Transformer bs1  | top1:81.2% top5:95.5% | top1:81.2% top5:95.5% | 210.9 fps  | 145.0 fps |
| Swin-Transformer bs16 | top1:81.2% top5:95.5% | top1:81.2% top5:95.5% | 432.3 fps | 170.7 fps  |
