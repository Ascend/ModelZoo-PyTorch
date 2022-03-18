环境准备：  

1.数据集路径  
通用的数据集统一放在/root/datasets/或/opt/npu/  
本模型数据集放在/root/datasets/  

2.进入工作目录  
cd MobileNetV3_large_100  

3.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
pip3.7 install -r requirements.txt 

4.获取模型代码  

git clone https://github.com/rwightman/gen-efficientnet-pytorch.git

5.安装模型代码  

python3.7 -m pip install -e gen-efficientnet-pytorch  

6.获取权重文件  
wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth 

7.获取benchmark工具  

根据所用机器的架构，将相应的benchmark.{arch} 放在当前目录

8.310上执行，执行时确保device空闲  
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/root/datasets  
