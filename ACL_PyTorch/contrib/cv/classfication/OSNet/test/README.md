环境准备：  

1.数据集路径  
[Market1501数据集(百度网盘下载，提取码：me3q)](https://pan.baidu.com/s/1Nl8tMEvq-MwNGd1pG4_6bg)  
Market1501数据集放在/root/datasets/，并将数据集文件夹命名为market1501。

2.进入工作目录  
cd OSNet

3.安装必要的依赖  
pip3.7 install -r requirements.txt

4.获取模型代码  
git clone https://github.com/KaiyangZhou/deep-person-reid.git  
cd deep-person-reid

5.加载模型  
python3.7 setup.py develop 

6.获取权重文件  
[OSNet训练pth权重文件(google下载)](https://drive.google.com/file/d/1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA/view?usp=sharing)  
[OSNet训练pth权重文件(百度网盘下载，提取码：gcfe)](https://pan.baidu.com/s/1Xkwa9TCZss_ygkC8obsEMg)  
将权重文件osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth放于OSNet目录下    

7.获取benchmark工具  
将benchmark.x86_64 benchmark.aarch64放于OSNet目录下   

8.310上执行，执行时确保device空闲   
cd OSNet   
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/root/datasets
