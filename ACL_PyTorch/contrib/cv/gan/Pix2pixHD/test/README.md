环境准备：  

1.进入主目录  
```
cd Pix2pixHD
```

2.获取模型代码  
```
git clone https://github.com/NVIDIA/pix2pixHD.git  
cd pix2pixHD  
```

3.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```

4.修改模型代码。 
```
patch -p1 < ../pix2pixhd_npu.diff  
cd ..
```

5.获取权重文件  
Pix2pixHD代码仓作者提供了预训练的模型，因此下载该预训练模型latest_net_G.pth并将其放置在Pix2PixHD目录下,之后在pix2pixHD目录下创建./checkpoints/label2city_1024p/目录，并将latest_net_G.pth移动至./pix2pixHD/checkpoints/label2city_1024p/目录下。
```
cd pix2pixHD
mkdir ./checkpoints/label2city_1024p/
mv ../latest_net_G.pth ./checkpoints/label2city_1024p/
cd ..
```

6.获取benchmark工具  
将benchmark.x86_64 benchmark.aarch64放在当前目录  

7.310上执行，执行时确保device空闲  
```
bash test/pth2om.sh   
bash test/eval_acc_perf.sh 
```

8.本推理GPU部分采用在线推理，故需要装有T4的gpu上安装运行pix2pixHD所需环境，在T4上获取模型代码。  

```
git clone https://github.com/NVIDIA/pix2pixHD.git  
cd pix2pixHD  
```
9.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```
10.修改模型代码。 
```
patch -p1 < ../pix2pixhd_gpu.diff  
cd ..
```

11.获取权重文件  
Pix2pixHD代码仓作者提供了预训练的模型，因此需要下载该预训练模型latest_net_G.pth,之后在pix2pixHD目录下创建./checkpoints/label2city_1024p/目录，并将latest_net_G.pth移动至./pix2pixHD/checkpoints/label2city_1024p/目录下。
```
cd pix2pixHD
mkdir ./checkpoints/label2city_1024p/
mv ../latest_net_G.pth ./checkpoints/label2city_1024p/
```

12.T4上采用预训练模型生成图片,此时生成的图片保存在./pix2pixHD/results/label2city_1024p/test_latest/images
```
bash ./scripts/test_1024p.sh
cd ..
```
13.精度对比
npu离线推理生成的图片保存在./Pix2pixHD/generated目录下
T4在线推理生成的图片保存在T4机器上./pix2pixHD/results/label2city_1024p/test_latest/images目录下
在两个位置存放了15个不同输入下，npu和gpu生成的结果，经过人工比对，相同输入下npu与gpu生成的图片完全一致，故而精度达标。

14.性能对比
npu离线推理的性能数据保存在./Pix2pixHD/result/perf_vision_batchsize_1_device_0.txt,该文件中Interface throughputRate一项的数值即为npu离线推理一张图片所耗时间。
gpu在线推理数据在执行12步后终端会给出每一次图片生成所耗时间，将15张图片生成的时间求和取平均即得到gpu上的性能数据。
两者进行对比，npu的性能数据高于gpu性能数据的一般，故性能达标。
