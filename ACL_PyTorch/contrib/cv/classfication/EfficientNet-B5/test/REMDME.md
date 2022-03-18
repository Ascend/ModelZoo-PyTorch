环境准备：

1.数据集路径
通用的数据集统一放在/root/datasets/或/opt/npu/
本模型数据集放在/opt/npu/
#环境准备：

##1.数据集路径
- 数据集统一放在/root/datasets/或/opt/npu/
- 本模型数据集放在/opt/npu/

##2.进入工作目录
```
cd EfficientNet-B5
```

##3.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装
```
pip3.7 install -r requirements.txt
```

##4.获取模型代码并将其中的/pycls/configs/dds_baselines/effnet/EN-B5_dds_8gpu.yaml文件复制到EfficientNet-B5文件夹中并改名efficientnetb5_dds_8gpu.yaml
```
git clone https://github.com/facebookresearch/pycls
cp ./pycls/configs/dds_baselines/effnet/EN-B5_dds_8gpu.yaml .
mv EN-B5_dds_8gpu.yaml efficientnetb5_dds_8gpu.yaml
```



##5.获取权重文件并改名为efficientnetb5.pyth
```
wget https://dl.fbaipublicfiles.com/pycls/dds_baselines/161305138/EN-B5_dds_8gpu.pyth
mv EN-B5_dds_8gpu.pyth efficientnetb5.pyth
```
##6.获取benchmark工具，将benchmark.x86_64放在当前目录


##7.310上执行，执行时确保device空闲
```
bash test/pth2om.sh
bash test/eval_acc_perf.sh --datasets_path=/opt/npu/
```
