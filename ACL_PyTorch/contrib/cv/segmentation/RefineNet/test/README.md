环境准备：  

1.进入主目录  

```
cd RefineNet
```

2.获取模型代码并通过patch进行修改

```bash
git clone https://github.com/DrSleep/refinenet-pytorch.git RefineNet_pytorch
cd RefineNet_pytorch
git am --signoff < ../RefineNet.patch
cd ..
```

3.安装必要的依赖。如果服务器上git clone太慢，可以在本地clone好后scp上传过去）

```bash
pip3 install -r requirements.txt
git clone https://github.com/drsleep/densetorch.git
cd densetorch
pip3 install -e .
```

4.获取数据集  
下载VOC2012数据集http://host.robots.ox.ac.uk/pascal/VOC/voc2012/，把VOCdevkit文件夹放到/opt/npu目录下

5.获取权重文件  
采用910训练的权重文件，该权重文件精度为0.7861，从Ascend modelzoo 本模型目录下载权重，把它放到RefineNet里的model/目录下

6.获取benchmark工具  
将benchmark.x86_64 benchmark.aarch64放在当前目录  

7.310上执行，执行时确保device空闲  

```bash
bash test/pth2om.sh
bash test/eval_acc_perf.sh --datasets_path=/opt/npu
```

