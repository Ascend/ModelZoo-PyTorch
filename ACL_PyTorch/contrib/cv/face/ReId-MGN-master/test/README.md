环境准备：

1.数据集路径
- [Market1501数据集获取路径](https://pan.baidu.com/s/1ntIi2Op?_at_=1624593258681)
- 原始数据集已经放在/opt/npu/Market1501/下，应在./ReID-MGN-master/data/下新建Market1501目录,将/opt/npu/Market1501/下的文件拷贝到./ReID-MGN-master/data/Market1501下
- ./data/Market1501/路径下，需要新建bin_data和bin_data_flip两个路径，bin_data和bin_data_flip两个路径下分别新建q和g两个路径
- 需要新建model路径，预训练文件model.pt放在该路径下
- 具体命令参考下文


2.进入工作目录
```
cd /ReID-MGN-master
mkdir -p ./data/Market1501
cp -r /opt/npu/Market1501/* ./data/Market1501/
mkdir -p ./data/Market1501/bin_data/q
mkdir -p ./data/Market1501/bin_data/p
mkdir -p ./data/Market1501/bin_data_flip/q
mkdir -p ./data/Market1501/bin_data_flip/p
mkdir model
```

3.安装必要的依赖
```
pip3.7 install -r requirements.txt
```

4.获取模型代码
```
git clone https://github.com/GNAYUOHZ/ReID-MGN.git MGN
cd MGN && git checkout f0251e9e6003ec6f2c3fbc8ce5741d21436c20cf && cd -
patch -R MGN/data.py < module.patch
```

5.获取权重文件
```
(https://pan.baidu.com/s/12AkumLX10hLx9vh_SQwdyw) password:mrl5
cp ${model.pt} ./model
```

6.获取benchmark工具
```
将benchmark.x86_64放在当前目录
```

7.310上执行，执行时确保device空闲
```
source env.sh
apt install dos2unix
dos2unix test/pth2om.sh
bash test/pth2om.sh
dos2unix test/eval_acc_perf.sh
bash test/eval_acc_perf.sh
```
