环境准备：

1.数据集路径
通用的数据集统一放在/root/datasets/或/opt/npu/
本模型数据集放在/opt/npu/

2.进入工作目录

```
cd Wide_ResNet101_2
```

3.安装必要的依赖

```
pip3.7 install -r requirements.txt
```

4.获取模型代码

```
git clone https://github.com/pytorch/vision
```

5.如果模型代码需要安装，则安装模型代码

```
cd vision
git reset 7d955df73fe0e9b47f7d6c77c699324b256fc41f --hard
python3.7 setup.py install
cd ..
```

6.获取权重文件

```
wget https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth
```

7.获取benchmark工具
将benchmark.x86_64 放在当前目录

8.310上执行，执行时确保device空闲

```
bash test/pth2om.sh
bash test/eval_acc_perf.sh --datasets_path=/opt/npu/
```
