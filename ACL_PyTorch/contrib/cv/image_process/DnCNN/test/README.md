环境准备：

1.获取数据集  
```
git clone https://github.com/SaoYan/DnCNN-PyTorch
```
开源代码仓的data目录下有数据集，将data复制到DnCNN目录下

2.进入工作目录  
```
cd DnCNN
```

3.安装必要的依赖  
```
pip3.7 install -r requirements.txt  
```

4.获取训练提供的权重文件
```
wget https://ascend-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/image_classification/DnCnn/net.pth  
```

5.获取benchmark工具  
将benchmark.x86_64 benchmark.aarch64放在当前目录  

6.310上执行，执行时确保device空闲
```
bash test/pth2om.sh
bash test/eval_acc_perf.sh  --datasets_path=./data
```
