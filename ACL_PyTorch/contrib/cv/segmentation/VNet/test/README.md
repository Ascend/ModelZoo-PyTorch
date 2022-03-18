环境准备：  

1.进入工作目录  
cd VNet  

2.获取模型代码  
```
git clone https://github.com/mattmacy/vnet.pytorch  
cd vnet.pytorch  
```

3.修改模型代码  
```
patch -p1 < ../vnet.patch  
```

4.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```

5.获取数据集  
从 https://luna16.grand-challenge.org/Download/ 下载subset0.zip~subset9.zip，解压到luna16/lung_ct_image目录下。并下载seg-lungs-LUNA16.zip，解压到luna16/seg-lungs-LUNA16目录下。之后执行原代码仓自带的预处理脚本。  
```
python normalize_dataset.py ./luna16 2.5 128 160 160  
```

6.获取权重文件  
暂无

7.获取benchmark工具  
将benchmark.x86_64 benchmark.aarch64放在当前目录  

8.310上执行，执行时确保device空闲  
```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh  
```

9.在t4环境上将onnx文件与perf_t4.sh放在同一目录  
然后执行bash perf_t4.sh，执行时确保gpu空闲  
