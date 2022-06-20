# 				BSN推理说明



## 1、 环境说明

1、安装必要的依赖

```
apex              0.1+ascend.20210930
certifi           2021.10.8
cycler            0.11.0
decorator         5.1.0
docutils          0.18
flatbuffers       2.0
future            0.18.2
Geohash           1.0
Hydra             2.5
kiwisolver        1.3.2
matplotlib        3.4.3
mpmath            1.2.1
numpy             1.21.0
onnx              1.10.2
onnxruntime       1.9.0
pandas            1.3.4
Pillow            8.4.0
pip               21.3.1
protobuf          3.19.1
pyparsing         3.0.6
python-dateutil   2.8.2
pytz              2021.3
scipy             1.7.2
setuptools        58.0.4
six               1.16.0
sympy             1.9
torch             1.5.0+ascend.post3.20210930
typing-extensions 3.10.0.2
wheel             0.37.0
```

2、获取开源代码

直接从githup上git clone 可能无法clone成功，建议先把githup上的仓先导入到git,再clone

```
git clone https://github.com/wzmsltw/BSN-boundary-sensitive-network.pytorch
```



3、获取onnx_tools,优化TEM的onnx模型

```
git clone https://gitee.com/zheng-wengang1/onnx_tools
```



4、下载视频特征数据集

请参考源代码仓

5、代码目录

```
BSN                     #模型名称命名的文件夹 
├── BSN-boundary-sensitive-network.pytorch   #BSN开源代码
	└── data
			├── activitynet_feature_cuhk
				├── csv_mean_100			#下载数据特征集
├── env.sh                      			#环境变量   
├── BSN_tem_pth2onnx.py     				#tem模型转换脚本
├── BSN_pem_pth2onnx.py     				#pem模型转换脚本
├── BSN_tem_preprocess.py   				#tem模型前处理脚本
├── BSN_pem_preprocess.py   				#pem模型前处理脚本
├── gen_dataset_info.py         			#生成数据集info文件   
├── BSN_tem_postprocess.py  				#tem模型后处理脚本
├── BSN_pem_postprocess.py  				#pem模型后处理脚本
├── BSN_eval.py  							#测试精度脚本
├── TEM_onnx_conv1d2conv2d.py  				#tem模型onnx，conv1d算子转conv2d算子优化脚本
├── requirements.txt            			#模型离线推理用到的所有且必要的依赖库  
├── README.md                   			#模型离线推理说明README     
├── modelzoo_level.txt          			#模型精度性能结果
└── test  
    ├── pth2om.sh  
    ├── eval_acc_perf.sh  
    ├── parse.py    
```



## 2、离线推理

1、pth权重转onnx



TEM的pth权重转onnx,参数pth_path为TEM模型权重文件所在位置，onnx_path为输出的onnx模型位置

```
python BSN_tem_pth2onnx.py --pth_path './tem_best.pth.tar' --onnx_path './BSN_tem.onnx'
```

tem-onnx模型优化,第一个参数为原本onnx模型位置，第二个参数为输出onnx模型

```
python TEM_onnx_conv1d2conv2d.py './BSN_tem.onnx' './BSN_tem1.onnx'
```

PEM的pth权重转onnx,参数pth_path为PEM模型权重文件所在位置，onnx_path为输出的onnx模型位置

```
python BSN_pem_pth2onnx.py --pth_path './pem_best.pth.tar' --onnx_path './BSN_pem.onnx'
```



2、onnx模型转om

使用atc工具将onnx模型转为om模型，注意应当先设置环境变量

```
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest

atc --framework=5 --model=BSN_tem1.onnx --output=BSN_tem_bs1 --input_format=ND --input_shape="video:1,400,100" --log=debug --soc_version=Ascend310

atc --framework=5 --model=BSN_pem.onnx --output=BSN_pem_bs1 --input_format=ND --input_shape="video_feature:1,1000,32" --log=debug --soc_version=Ascend310
```



3、TEM推理

运行预处理脚本,运行前确保你已经clone了开源代码，并下载数据特征集

```
python BSN_tem_preprocess.py
```

获取处理数据集信息，第一个参数为模型类型，第二个参数为特征文件位置，第三个参数为输出文件名，第四、五个参数为特征形状（400*100）

```
python gen_dataset_info.py tem /home/wch/BSN/BSN-TEM-preprocess/feature TEM-video-feature 400 100
```

使用benchmark工具进行推理,生成的数据文件会放在当前路径的result/dumpOutput_device0目录下

```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=BSN_tem_bs1.om -input_text_path=./TEM-video-feature.info -input_width=400 -input_height=100 -output_binary=True -useDvpp=False
```

使用BSN_tem_postprocess.py进行tem后处理(tem的后处理与pem的前处理有关请按照顺序执行)

```
python BSN_tem_postprocess.py  --TEM_out_path ./result/dumpOutput_device0
```



4、PEM推理

运行pem预处理脚本(pem的前处理与tem的后处理有关请按照顺序执行)

```
python BSN_pem_preprocess.py
```

获取处理数据集信息，第一个参数为模型类型，第二个参数为特征文件位置，第三个参数为输出文件名，第四、五个参数为特征形状（1000*32）

```
python get_info.py pem output/BSN-PEM-preprocess/feature PEM-video-feature 1000 32
```

使用benchmark工具进行推理,生成的数据文件会放在当前路径的result/dumpOutput_device1目录下

```
./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=1 -om_path=BSN_pem_bs1.om -input_text_path=./PEM-video-feature.info -input_width=1000 -input_height=32 -output_binary=True -useDvpp=False
```

运行后处理脚本,会在output目录下生成结果文件

```
python BSN_pem_postprocess.py --PEM_out_path result/dumpOutput_device1
```



5、精度测试

原本代码仓的代码是python2的代码，在在使用前需要转为python3

```
2to3 -w ./BSN-boundary-sensitive-network.pytorch/Evaluation/eval_proposal.py

```

精度测试

```
python BSN_eval.py
```



6、整体测试

运行脚本，直接转om模型

```
bash ./test/pth2om.sh
```

运行脚本，进行离线推理,运行脚本前，请确保已经将源代码中使用的文件，转为python3

```
bash ./test/eval_acc_perf.sh
```



## 3 精度性能对比

### 1、精度对比

​	pth精度

```
Model   论文    			开源pth文件			离线推理精度
BSN   	AR100：72.42      74.34				74.34
```

### 2、性能对比

#### 2.1 npu性能数据

tem bs1性能数据

```
-----------------Performance Summary------------------
[e2e] throughputRate: 180.879, latency: 106303
[data read] throughputRate: 182.039, moduleLatency: 5.49332
[preprocess] throughputRate: 181.865, moduleLatency: 5.49859
[inference] throughputRate: 182, Interface throughputRate: 3275.55, moduleLatency: 0.561457
[postprocess] throughputRate: 182.009, moduleLatency: 5.49425

-----------------------------------------------------------
```

pem bs1性能数据

```
-----------------Performance Summary------------------
[e2e] throughputRate: 616.804, latency: 7665.32
[data read] throughputRate: 1840.06, moduleLatency: 0.54346
[preprocess] throughputRate: 1817.62, moduleLatency: 0.550169
[inference] throughputRate: 1839.62, Interface throughputRate: 3874.46, moduleLatency: 0.469866
[postprocess] throughputRate: 1839.86, moduleLatency: 0.543521

-----------------------------------------------------------
```

tem单卡吞吐率：3275.55x4=13102.2

pem单卡吞吐率：3874.46x4=15497.84

BSN整体吞吐率为：1/（1/13102.2+1/15497.84）=7099.87



tem bs16性能数据

```
-----------------Performance Summary------------------
[e2e] throughputRate: 143.161, latency: 134310
[data read] throughputRate: 144.544, moduleLatency: 6.91832
[preprocess] throughputRate: 144.393, moduleLatency: 6.92554
[inference] throughputRate: 144.476, Interface throughputRate: 12277.9, moduleLatency: 0.570148
[postprocess] throughputRate: 9.03906, moduleLatency: 110.631

-----------------------------------------------------------
```

pem bs16性能数据

```
-----------------Performance Summary------------------
[e2e] throughputRate: 141.751, latency: 33354.2
[data read] throughputRate: 145.216, moduleLatency: 6.88627
[preprocess] throughputRate: 144.936, moduleLatency: 6.89961
[inference] throughputRate: 145.023, Interface throughputRate: 18564.9, moduleLatency: 0.483157
[postprocess] throughputRate: 9.10977, moduleLatency: 109.772

-----------------------------------------------------------
```

tem单卡吞吐率：12277.9x4=49111.6

pem单卡吞吐率：18564.9x4=74259.6

BSN整体吞吐率为：1/（1/49111.6+1/74259.6）=29560.95

#### 2.2 T4性能数据

在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2 

batch1性能：

tem:

```
trtexec --onnx=BSN_tem.onnx --fp16 --shapes=video:1*400*100 --threads
```



```
[11/23/2021-06:45:38] [I] GPU Compute
[11/23/2021-06:45:38] [I] min: 0.045166 ms
[11/23/2021-06:45:38] [I] max: 2.00708 ms
[11/23/2021-06:45:38] [I] mean: 0.0565804 ms
[11/23/2021-06:45:38] [I] median: 0.0568848 ms
[11/23/2021-06:45:38] [I] percentile: 0.0620117 ms at 99%
[11/23/2021-06:45:38] [I] total compute time: 2.47115 s
```

pem:

```
trtexec --onnx=BSN_pem.onnx --fp16 --shapes=video:1*1000*32 --threads
```



```
[11/19/2021-06:40:06] [I] GPU Compute
[11/19/2021-06:40:06] [I] min: 0.0185547 ms
[11/19/2021-06:40:06] [I] max: 1.26123 ms
[11/19/2021-06:40:06] [I] mean: 0.0205523 ms
[11/19/2021-06:40:06] [I] median: 0.0201416 ms
[11/19/2021-06:40:06] [I] percentile: 0.0458527 ms at 99%
[11/19/2021-06:40:06] [I] total compute time: 0.793032 s
```



tem单卡吞吐率：1000/0.215458=17674

pem单卡吞吐率：1000/0.0205523=48656

BSN单卡吞吐率：1000/（0.215458+0.0205523）=12965





batch16性能：

tem:

```
trtexec --onnx=BSN_tem.onnx --fp16 --shapes=video:16*400*100 --threads
```



```
[11/19/2021-06:37:12] [I] GPU Compute
[11/19/2021-06:37:12] [I] min: 0.182129 ms
[11/19/2021-06:37:12] [I] max: 0.252548 ms
[11/19/2021-06:37:12] [I] mean: 0.219561 ms
[11/19/2021-06:37:12] [I] median: 0.218262 ms
[11/19/2021-06:37:12] [I] percentile: 0.245422 ms at 99%
[11/19/2021-06:37:12] [I] total compute time: 1.5714 s
```

pem:

```
trtexec --onnx=BSN_pem.onnx --fp16 --shapes=video:16*1000*32 --threads
```



```
[11/23/2021-06:51:29] [I] GPU Compute
[11/23/2021-06:51:29] [I] min: 0.21167 ms
[11/23/2021-06:51:29] [I] max: 2.40039 ms
[11/23/2021-06:51:29] [I] mean: 0.24159 ms
[11/23/2021-06:51:29] [I] median: 0.240479 ms
[11/23/2021-06:51:29] [I] percentile: 0.25769 ms at 99%
[11/23/2021-06:51:29] [I] total compute time: 2.08734 s
```

tem单卡吞吐率：1000/（0.219561/16）=72872

pem单卡吞吐率：1000/（0.24159/16）=66228

BSN单卡吞吐率：1000/（（0.219561+0.0210533）/16）=34696



#### 2.3 性能对比

batch1 :

​	TEM

​		310:13102

​		T4:17674

​	PEM:

​		310:15498

​		T4:48656

​	BSN:

​		7099.87<12965

​		7099.87/12965=0.548

batch16:

​	TEM:

​			310: 49111.6

​			t4:	72872

​	PEM:

​			310: 74259.6

​			T4： 66228

​	BSN:

​		29560.95<34696

​		29560.95/34696=0.85

在batch1，310性能是0.548倍T4性能；在batch16,310性能是0.85倍T4性能。

