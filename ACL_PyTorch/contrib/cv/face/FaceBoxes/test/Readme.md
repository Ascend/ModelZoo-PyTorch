环境准备：

1.数据集路径
- FDDB数据集获取路径](https://drive.google.com/open?id=17t4WULUDgZgiSy5kpCax4aooyPaz3GQH)，放在目录 /opt/npu/FDDB 下
- 预训练文件获取路径(https://drive.google.com/file/d/1tRVwOlu0QtjvADQ2H7vqrRwsWEmaqioI)，放在目录 ./weights下
- 具体命令参考下文


2.进入工作目录
```
cd /FaceBoxes
mkdir -p ./data/FDDB
mkdir -p ./weights
```

3.安装必要的依赖
```
pip3.7 install -r requirements.txt
```

4.获取模型代码
```
git clone https://gitee.com/charles_licheng/faceboxes.git
```

5.获取benchmark工具
```
将benchmark工具放在当前目录
```

6.310上执行，执行时确保device空闲
```
apt install dos2unix
dos2unix test/pth2om.sh
bash test/pth2om.sh
dos2unix test/eval_acc_perf.sh
bash test/eval_acc_perf.sh --datasets_path=/opt/npu/FDDB
```
7.在t4环境上将onnx文件与perf.sh放在同一目录  

-执行 bash perf_g.sh, 执行时确保gpu空闲