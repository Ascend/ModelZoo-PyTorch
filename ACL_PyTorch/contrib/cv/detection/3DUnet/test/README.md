环境准备：

1.数据集路径
数据集统一放在/root/datasets/或/opt/npu/
本模型数据集放在/root/datasets/

2.进入工作目录
cd 3DUnet

3.安装必要的依赖
pip3.7 install -r requirements.txt

4.获取，修改与安装开源模型代码 git clone https://github.com/black0017/MedicalZooPytorch

5.获取权重文件
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/segmentation/3DUnet/UNET3D.pth

6.获取msame工具
将msame放在当前目录

7.310上执行，执行时确保device空闲
python3 inference.py

8.在t4环境上获取模型代码
然后执行python3 inference.py，执行时确保gpu空闲