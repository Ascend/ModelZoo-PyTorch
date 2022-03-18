环境准备：  

1.数据集路径  
数据集统一放在/root/datasets/或/opt/npu/  
本模型数据集放在/root/datasets/  

2.进入工作目录  
cd DPN131 

3.安装必要的依赖  
pip3.7 install -r requirements.txt 

4.获取，修改与安装开源模型代码
git clone https://github.com/Cadene/pretrained-models.pytorch.git
cd pretrained-models.pytorch
patch -p1 < ../dpn.diff,其中dpn.diff是通过git diff > ./dpn.diff生成的  
如果模型代码需要安装，则安装模型代码(如果没有安装脚本，pth2onnx等脚本需要引用模型代码的类或函数，可通过sys.path.append(r"./pretrained-models.pytorch")添加搜索路径的方式)
cd ..

5.获取权重文件  
wget http://data.lip6.fr/cadene/pretrainedmodels/dpn131-7af84be88.pth

6.获取benchmark工具  
将benchmark.x86_64放在当前目录  

7.310上执行，执行时确保device空闲  
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/root/datasets  
