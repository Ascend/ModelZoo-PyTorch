环境准备：  

1.数据集路径  
数据集统一放在/root/datasets/或/opt/npu/  
本模型数据集放在/opt/npu/  

2.进入工作目录  
cd pnasnet5large  

3.安装必要的依赖  
pip3.7 install -r requirements.txt 

4.获取模型代码  
git clone https://github.com/rwightman/pytorch-image-models.git 

5.如果模型代码需要安装，则安装模型代码  
cd pytorch-image-models
python3.7 setup.py install  
cd ..  

6.获取权重文件  
脚本自动下载

7.获取benchmark工具  
将benchmark.x86_64放在当前目录  

8.310上执行，执行时确保device空闲  
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/opt/npu/  

9.在t4环境上将onnx文件与perf_t4.sh放在同一目录  
然后执行bash perf_t4.sh，执行时确保gpu空闲  