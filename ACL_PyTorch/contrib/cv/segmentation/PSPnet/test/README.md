环境准备：  

1.数据集路径  
数据集统一放在/root/datasets/或/opt/npu/  
本模型数据集放在/opt/npu/ 

2.进入工作目录  
cd pspnet  

3.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
pip3.7 install -r requirements.txt

4.获取、修改和安装模型代码  
pip3.7 install mmcv-full==1.3.10
git clone https://github.com/open-mmlab/mmsegmentation.git  
cd mmsegmentation  
pip3.7 install -e .  # or "python3.7 setup.py develop"  
cd ..  
 
5.获取权重文件  
wget https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x512_20k_voc12aug/pspnet_r50-d8_512x512_20k_voc12aug_20200617_101958-ed5dfbd9.pth   

6.获取benchmark工具  
将benchmark.x86_64放在当前目录  

7.310上执行，执行时确保device空闲  
bash test/pth2om.sh  
bash test/eval_acc_perf.sh   

8.在t4环境上将onnx文件与perf_t4.sh放在同一目录  
然后执行bash perf_t4.sh，执行时确保gpu空闲  
