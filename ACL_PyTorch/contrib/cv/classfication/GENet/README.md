## <a name="1">1. 模型概述</a>
### 1.1 参考论文
[GENet论文](https://arxiv.org/abs/1810.12348)
### 1.2 参考实现
[代码地址](https://github.com/BayesWatch/pytorch-GENet)
> branch: master

> commit id: 3fbf99fb6934186004ffb5ea5c0732e0e976d5b2

## <a name="1">2. 推理环境准备</a>
### 2.1 环境介绍
CANN=[5.0.3](https://www.hiascend.com/software/cann/commercial?version=5.0.3)。  
硬件环境、开发环境和运行环境准备请参见[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/504/envdeployment/instg)。
### 2.2 所需依赖
```
Pytorch==1.5.0
Torchvision==0.6.0
ONNX==1.7.0
numpy==1.18.5
Pillow==7.2.0
```
### 2.3 环境配置
```
pip3.7 install -r requirements.txt  
```
## <a name="1">3. 数据集准备</a>
### 3.1 下载数据集
在官方[链接](http://www.cs.toronto.edu/~kriz/cifar.html)下载cifar10数据集
### 3.2 数据预处理
准备Bin文件  
```
python3.7 preprocess.py ${datasets_path} ./prep_dataset 
```
第一个参数为数据集存放目录（例：若数据集路径为/home/HwHiAiUser/dataset/cifar-10-batches-py/，则数据集存放目录为/home/HwHiAiUser/dataset/），第二个参数为预处理后的数据文件的相对路径。该操作会在数据文件的目录下生成标签文件val_label.txt。
### 3.3 生成数据集info文件
```
python3.7 get_info.py bin ./prep_dataset ./genet_prep_bin.info 32 32
```
第一个参数为生成的数据集文件格式，第二个参数为预处理后的数据文件的相对路径，第三个参数为生成的数据集文件保存的路径。运行成功后，在当前目录中生成genet_prep_bin.info。
## <a name="1">4. 模型转换</a>
### 4.1 获取源码
```
git clone https://github.com/BayesWatch/pytorch-GENet.git 
cd pytorch-GENet/
git reset 3fbf99fb6934186004ffb5ea5c0732e0e976d5b2 --hard
cd ../
```
### 4.2 pth转onnx模型
```
python3.7 pthtar2onnx.py ${model_path}
```
其中${model_path}指的是模型路径，如/home/HwHiAiUser/model/genet.pth.tar
### 4.3 onnx转om模型
设置环境变量
```
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
```
使用ATC工具转换，工具使用方法可以参考[《CANN 开发辅助工具指南 (推理)》](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)
```
bash test/onnx2om.sh
```
该指令会生成genet_bs16_tuned，genet_bs1_tuned两个模型，可在onnx2om.sh文件中修改以生成不同bs值的om模型
>  **说明**
> 注意目前ATC支持的onnx算子版本为11
## <a name="1">5. 推理验证</a>
### 5.1 使用指南
[《CANN 推理benchmark工具用户指南》](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)
### 5.2 离线推理
1.增加benchmark.{arch}可执行权限
```
chmod u+x benchmark.x86_64
```
2.推理
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=genet_bs1_tuned.om -input_text_path=genet_prep_bin.info -input_width=32 -input_height=32 -output_binary=False -useDvpp=False
```
可以通过修改batch_size的值进行在不同batchsize情况下的推理（同时要修改对应的om文件路径即om_path的值）。运行该指令后输出结果默认保存在当前目录/result/dumpOutput_device0中，同时在/result目录下会生成一个推理性能文件  

3.获取性能信息
```
python3.7 test/parse.py result/perf_vision_batchsize_1_device_0.txt
```
运行该指令获得bs为1时推理所得的310的性能信息，实例如下：
```
[e2e] throughputRate: 132.777, latency: 75314.4
[data read] throughputRate: 134.331, moduleLatency: 7.44429
[preprocess] throughputRate: 134.114, moduleLatency: 7.45634
[inference] throughputRate: 134.244, Interface throughputRate: 809.813, moduleLatency: 1.35099
[postprocess] throughputRate: 134.257, moduleLatency: 7.44838
```
4，gpu设备的推理
将onnx模型置于装有gpu的设备，参考以下指令获取onnx模型在gpu上推理的性能信息
```
bash test/perf_g.sh
```
## <a name="1">6. 精度验证</a>
1.参考以下指令生成精度信息文件
```
python3.7 cifar10_acc_eval.py result/dumpOutput_device0/ ./prep_dataset/val_label.txt ./ result_bs1.json
```
第一个参数为生成推理结果所在路径，第二个参数为标签数据，第三个参数为生成结果文件路径，第四个参数为生成结果文件名  

2.参考以下指令获取310上推理的精度信息
```
python3.7 test/parse.py result_bs1.json
```
精度参考：  
|  GENET模型|  gpu吞吐率| 310吞吐率 |  精度|
|--|--|--|--|
|  bs1|  1805.315fps| 3239.252fps|Error@1 5.78 Error@5 0.15|
|  bs16| 5922.109fps| 7796.88fps|Error@1 5.78 Error@5 0.15 |


## <a name="7">7. 性能对比</a>
测试时要保证设备空闲，npu-smi info可以查看设备状态。benchmark工具在整个数据集上推理方式测性能可能时间较长，纯推理方式测性能可能不准确，因此bs1与bs16要使用在整个数据集上推理的方式测性能，bs4、8、32可以用纯推理的方式测性能。benchmark工具测的Interface throughputRate或samples/s数据是单个device吞吐率，计算310单卡吞吐率需要乘以4。tensorrt工具测的t4数据GPU Compute的mean代表batch个数据的时延，1000/(GPU Compute mean/batch)可以将其转换为吞吐率。  
### 7.1 310性能数据
以310的bs1为例:
```
[e2e] throughputRate: 132.777, latency: 75314.4
[data read] throughputRate: 134.331, moduleLatency: 7.44429
[preprocess] throughputRate: 134.114, moduleLatency: 7.45634
[inference] throughputRate: 134.244, Interface throughputRate: 809.813, moduleLatency: 1.35099
[postprocess] throughputRate: 134.257, moduleLatency: 7.44838
```
Interface throughputRate: 809.813，809.813x4=3239.252 fps。即是batch1 310单卡吞吐率,batch16的计算方法同理
> 为了避免长期占用device， bs4,8,32使用纯推理测性能，其中，对bs4进行纯推理输入命令如下所示，其中batch_size=4表示bs的值，在对不同bs值对应的om模型进行推理时需要做出相应的更改：
> `./benchmark.x86_64 -device_id=0 -om_path=genet_bs4_tuned.om -round=30 -batch_size=4`
> 
计算bs4,8,32的吞吐率时，计算方法也同样为Interface throughputRate乘4
### 7.2 710性能数据
710的推理过程与310相似，可以参考前面310的步骤。不同的是710的吞吐率即为Interface throughputRate的值，无需乘4
### 7.3 T4性能数据
在运行5.2的gpu推理指令时，我们会获得相关的性能信息，以T4的bs1为例：
```
[05/13/2022-16:53:32] 
[I] GPU Compute Time: 
min = 0.512207 ms, 
max = 2.97815 ms, 
mean = 0.55392 ms, 
median = 0.540283 ms,
percentile(99%) = 0.690048 ms
```
batch1 t4单卡吞吐率：1000/(0.55392/1)=1805.315 fps  
计算方法为1000/(GPU Compute mean/batch)
### 7.4 性能对比
310、710、T4都取性能最优（即最优batch）的数据进行比较；  
710的最优batch性能 >=1.2倍310最优batch性能x 4且710的最优batch性能 >=1.6倍T4最优batch性能时性能达标