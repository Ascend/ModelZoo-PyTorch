环境准备：

1.数据集路径 通用的数据集统一放在/root/datasets/或/opt/npu/ 
本模型数据集放在/opt/npu/

2.进入工作目录
```
cd xception
```
3.安装必要的依赖
```
pip3.7 install -r requirements.txt
```
4.获取模型代码
```
git clone https://github.com/tstandley/Xception-PyTorch
```
5.5.如果使用补丁文件修改了模型代码则将补丁打入模型代码,如果需要引用模型代码仓的类或函数通过sys.path.append(r"./Xception-PyTorch")添加搜索路径。
```

cd Xception-PyTorch  
git reset 7b9718bb525fefc95f507306e685aa8998d0492c --hard  
cd ..
```
6.获取权重文件
```
https://ascend-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/image_classification/xception/pth/xception-c0a72b38.pth.tar
```
7.获取benchmark工具
将benchmark.x86_64 放在当前目录

8.310上执行，执行时确保device空闲
```
bash test/pth2om.sh
bash test/eval_acc_perf.sh --datasets_path=/opt/npu/
```
