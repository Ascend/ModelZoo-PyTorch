环境准备：  

1.数据集路径  
通用的数据集统一放在项目文件夹/root/datasets或/opt/npu/  
本模型数据集放在/data/widerface/val/images
```
ln -s /opt/npu/wider_face/WIDER_val/images/ data/widerface/val/images
```
2.获取模型代码
```
git clone https://github.com/biubug6/Pytorch_Retinaface 
cp Pytorch_Retinaface/utils . -rf
cp Pytorch_Retinaface/data . -rf
```

3.下载pth权重文件,放在项目根目录下
```
cd Pytorch_Retinaface
mkdir weights
cd weights
```
[Retinaface预训练pth权重文件 百度云](https://pan.baidu.com/s/12h97Fy1RYuqMMIV-RpzdPg) Password: fstq
将权重文件放在weights目录下

4.修改Pytorch_Retinaface/data/config.py,将cfg_mnet中的'pretrain'改为False,另外安装必要依赖
```
cd .. 
pip install -r requirements.txt
```
5.310上执行，执行时确保device空闲 ,生成om文件
```
bash test/pth2om.sh
```
6.获取benchmark工具  
将benchmark.x86_64 benchmark.aarch64放在根目录下
```
cp xxx/benchmark ./
```
7.评估精度
```
bash test/eval_acc.sh
```

8.在t4环境上将onnx文件与perf_t4.sh放在同一目录  
然后执行bash perf_t4.sh，执行时确保gpu空闲
```
bash perf_t4.sh
```
