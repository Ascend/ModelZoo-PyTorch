环境准备：  
  
1.数据集路径  
通用的数据集统一放在/root/datasets/或/opt/npu/  
本模型数据集放在/opt/npu/  
  
2.进入工作目录  
cd SENet  
  
3.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
pip3.7 install -r requirements.txt  
  
4.获取模型代码   
git clone [https://github.com/Cadene/pretrained-models.pytorch.git](https://github.com/Cadene/pretrained-models.pytorch.git)  
  
5.安装模型代码  
cd pretrained-models.pytorch  
python3.7 setup.py install  
cd ..  

6.获取权重文件  
wget [http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth](http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth)  

7.获取benchmark工具  
将benchmark.x86_64 benchmark.aarch64放在当前目录  
  
8.310上执行，执行时确保device空闲  
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/opt/npu  
