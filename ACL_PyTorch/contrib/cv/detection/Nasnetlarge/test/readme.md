环境准备：  

1.数据集路径  
通用的数据集统一放在/root/datasets/或/opt/npu/  
本模型数据集放在/opt/npu/  

2.进入工作目录  
cd nasnetlarge

3.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
pip3.7 install -r requirements.txt 

4.获取模型代码  
[代码地址](https://github.com/Cadene/pretrained-models.pytorch#nasnet)  
branch:master    
commit id：b8134c79b34d8baf88fe0815ce6776f28f54dbfe

5.本模型代码需要安装

cd pretrained-models.pytorch

python3.7 setup.py install

6.获取权重文件  
wget http://data.lip6.fr/cadene/pretrainedmodels/nasnetalarge-a1897284.pth

7.获取benchmark工具  
将benchmark.x86_64 benchmark.aarch64 放在当前目录
获取onnx优化工具：
cd test
git clone https://gitee.com/zheng-wengang1/onnx_tools.git ./onnx_tools
cd ..

8.310上执行，执行时确保device空闲  
bash test/pth2om.sh  
bash test/eval_acc_perf.sh 

9.在gpu环境上将onnx文件与perform_benchmark.sh放在同一目录  
然后执行bash perform_benchmark.sh，执行时确保gpu空闲  
