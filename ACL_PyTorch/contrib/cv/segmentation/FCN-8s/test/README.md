环境准备：  

1.数据集路径  
数据集统一放在/root/datasets/或/opt/npu/  
本模型数据集放在/opt/npu/ 

2.进入工作目录  
cd fcn-8s 

3.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
pip3.7 install -r requirements.txt

4.获取、修改和安装模型代码  
git clone https://github.com/open-mmlab/mmcv.git  
cd mmcv  
pip3.7 install -e .  
cd ..  
git clone https://github.com/open-mmlab/mmsegmentation.git  
cd mmsegmentation  
pip3.7 install -e .  # or "python3.7 setup.py develop"  
cd ..  

5.获取权重文件  
wget https://download.openmmlab.com/mmsegmentation/v0.5/fcn/fcn_r50-d8_512x512_20k_voc12aug/fcn_r50-d8_512x512_20k_voc12aug_20200617_010715-52dc5306.pth  

6.获取benchmark工具  
将将benchmark.x86_64,benchmark.aarch64放在当前目录  

7.310上执行，执行时确保device空闲  
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/opt/npu/
