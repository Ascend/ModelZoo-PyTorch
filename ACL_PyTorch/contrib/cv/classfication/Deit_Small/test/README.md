
环境准备：  

1.数据集路径  
通用的数据集统一放在/root/datasets/或/opt/npu/  
本模型数据集路径为 /opt/npu/

2.进入工作目录  
cd Deit-Small  

3.导入所需的环境
pip3.7 install -r requirements.txt

4.获取模型代码  
git clone https://github.com/facebookresearch/deit.git

5.获取权重文件  
wget https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth

6.获取benchmark工具  
将benchmark.x86_64 benchmark.aarch64放在当前目录  

7.310上执行，执行时确保device空闲  
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/root/datasets  
