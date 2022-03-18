环境准备：  

1.进入主目录  
```
cd Nested_UNet  
```

2.获取模型代码  
```
git clone https://github.com/4uiiurz1/pytorch-nested-unet  
cd pytorch-nested-unet
```

3.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt
```

4.修改模型代码  
```
patch -p1 < ../nested_unet.diff
```

5.获取数据集  
从 https://www.kaggle.com/c/data-science-bowl-2018/data 下载stage1_train.zip到pytorch-nested-unet目录下，将其解压到指定目录。之后执行原代码仓自带的预处理脚本，并将处理好的数据集复制到主目录下。  
```
mkdir -p inputs/data-science-bowl-2018/stage1_train/
unzip -d inputs/data-science-bowl-2018/stage1_train/ stage1_train.zip
python3.7 preprocess_dsb2018.py
cp -r inputs/dsb2018_96/ ../
cp val_ids.txt ../
```

6.获取权重文件  
由于原代码仓没有提供预训练的模型，因此需要自行在GPU环境下训练模型，之后将权重文件复制到主目录下。
```
python3.7 train.py --dataset dsb2018_96 --arch NestedUNet --loss LovaszHingeLoss --epochs 200  
cp models/dsb2018_96_NestedUNet_woDS/model.pth ../nested_unet.pth
cd ../
```

7.获取benchmark工具  
将benchmark.x86_64 benchmark.aarch64放在当前目录  

8.310上执行，执行时确保device空闲  
```
bash test/pth2om.sh
bash test/eval_acc_perf.sh
```
