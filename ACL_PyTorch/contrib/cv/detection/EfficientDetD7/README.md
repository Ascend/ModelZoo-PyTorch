# EfficientDet D7 

### 推理流程
#### 环境准备
python3.7 install -r requirement.txt
```
克隆本代码仓后
cd EfficientDetD7
git clone https://github.com/rwightman/efficientdet-pytorch.git
cd efficientdet-pytorch
git checkout c5b694aa34900fdee6653210d856ca8320bf7d4e
patch -p1 < ../effdet.patch

cd ..
git clone https://gitee.com/zheng-wengang1/onnx_tools.git
cd onnx_tools
git checkout cbb099e5f2cef3d76c7630bffe0ee8250b03d921

d7.pth权重文件获取
链接: https://pan.baidu.com/s/1rDt4I9yobrApJqQFS13n3A  密码: 1f47
下载完成后
mkdir model
mv EfficientDet_file/d7.pth ./model
rm -r EfficientDet_file
coco数据集下载地址：http://images.cocodataset.org/zips/val2017.zip
标注json文件下载地址：http://images.cocodataset.org/annotations/annotations_trainval2017.zip
下载完成后
unzip val2017.zip
unzip annotations_trainval2017.zip
mkdir coco_data
mv annotations coco_data
mv val2017 coco_data
coco_data目录结构需满足:
coco_data
    ├── annotations
    └── val2017
```

#### 导出onnx模型
注：d7模型参数及输入shape过大，只能进行batch_size=1的推理
* 1、运行pth2onnx.py
```
batch_size=1
python3.7 pth2onnx.py --batch_size=1 --checkpoint=./model/d7.pth --out=./model/d7_bs1.onnx 
```
参数解释：  
batch_size:根据需求选择batch大小
checkpoint：pytorch权重文件  
out：输出onnx模型  
* 2、精简优化网络
```
pip3 install onnx-simplifier
batch_size=1
python3.7 -m onnxsim --input-shape="1,3,1536,1536" --dynamic-input-shape ./model/d7_bs1.onnx ./model/d7_bs1_sim.onnx
```
* 3、部分pad算子constant_value值-inf修改为0
```
batch_size=1
python3.7 modify_onnx.py --model=./model/d7_bs1_sim.onnx --out=./model/d7_bs1_modify.onnx
```
#### 转为om模型
* 使用Ascend atc工具将onnx转换为om
```
CANN安装目录
source /usr/local/Ascend/ascend-toolkit/set_env.sh
将atc日志打印到屏幕
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
设置日志级别
#export ASCEND_GLOBAL_LOG_LEVEL=0 #debug 0 --> info 1 --> warning 2 --> error 3
开启ge dump图
#export DUMP_GE_GRAPH=2
参考命令
batch_size=1
atc --framework=5 --model=./model/d7_bs1_modify.onnx --output=./model/d7_bs1 --input_format=NCHW --input_shape="x.1:1,3,1536,1536" --log=debug --soc_version=Ascend310
```
说明：  

1.--input_shape是模型输入节点的shape，可使用netron查看onnx输入节点名与shape，batch维值为16，即会生成batch size为16的om模型。无论onnx模型的batch是多少，只要通过--input_shape指定batch为正整数，就得到对应batch size的om模型，om模型虽然支持动态batch，但是我们不使用动态batch的om模型  
2.--out_nodes选项可以指定模型的输出节点，形如--out_nodes="节点1名称:0;节点2名称:0;节点3名称:0"就指定了这三个节点每个节点的第1个输出作为模型的第一，第二，第三个输出  
3.算子精度通过参数--precision_mode选择，默认值force_fp16  
3.开启autotune方法：添加--auto_tune_mode="RL,GA"  
5.开启repeat autotune方法：添加--auto_tune_mode="RL,GA"同时export REPEAT_TUNE=True  
6.配置环境变量ASCEND_SLOG_PRINT_TO_STDOUT和ASCEND_GLOBAL_LOG_LEVEL，然后执行命令atc ... > atc.log可以输出日志到文件  
7.配置环境变量DUMP_GE_GRAPH后执行atc命令时会dump中间过程生成的模型图，使用华为修改的netron可以可视化这些.pbtxt模型文件，如需要请联系华为方，当atc转换失败时可以查看ge生成的中间过程图的模型结构与算子属性，分析出哪个算子引起的问题  
8.如果使用aipp进行图片预处理需要添加--insert_op_conf=aipp_efficientnet-b0_pth.config  
9.atc工具的使用可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01  
10.若模型包含atc不支持的算子，算子问题可以规避的先通过修改模型进行规避，并在modelzoo上提issue或联系华为方  

#### 测试集预处理
coco val 5k 数据集下载
```
cd EfficientDetD7
mkdir bin_save
python3.7 preprocess.py --root=coco_data --bin-save=bin_save
python3.7 get_info.py bin ./bin_save ./d7_bin.info 1536 1536
```
选择coco val数据集进行验证  
参数解释：  
root:coco 验证集的路径   
bin_save：处理完bin文件存放路径  
预处理后的数据集信息文件d0_bin.info:
```
0 ./bin_save/000000184384.bin 1536 1536  
1 ./bin_save/000000182805.bin 1536 1536  
2 ./bin_save/000000223182.bin 1536 1536  
...
```
第一列为样本序号，第二列为预处理后的样本路径，第三四列为预处理后样本的宽高

#### 离线推理

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.1 推理benchmark工具用户指南 01
将benchmark工具放置于当前目录，与model同级
* 二进制输入
```
batch_size=1
./benchmark.x86_64 -model_type=vision -batch_size=1 -device_id=1 -om_path=./model/d7_bs1.om -input_text_path=d7_bin.info -input_width=1536 -input_height=1536 -output_binary=True -useDvpp=False
```

说明：
-model_type为benchmark支持的模型类型，目前支持的有vision，nmt，widedeep，nlp，yolocaffe，bert，deepfm  
-device_id是指运行在ascend 310的哪个device上，每张ascend 310卡有4个device  
-batch_size是指om模型的batch大小，该值应与om模型的batch大小相同，否则报输入大小不一致的错误  
-om_path是om模型文件路径  
-input_text_path为包含数据集每个样本的路径与其相关信息的数据集信息文件路径  
-input_height为输入高度  
-input_width为输入宽度  
-output_binary为以预处理后的数据集为输入，benchmark工具推理om模型的输出数据保存为二进制还是txt，但对于输出是int64类型的节点时，指定输出为txt时会将float类型的小数转换为0而出错  
-useDvpp为是否使用aipp进行数据集预处理  

#### 测试数据后处理
```
python3..7 postprocess.py --root=./coco_data --omfile=./result/dumpOutput_device1
```
参数解释：  
root：测试集目录   
omfile:om推理出的数据存放路径  

## EfficientDet-D7测试结果:

#### 性能测试
```
batch_size=1
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_d7_bs1_modify_in_device_1.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 1.50578samples/s, ave_latency: 664.142ms

GPU在线推理
cd efficientdet-pytorch 
python3.7 validate.py ./coco_data --model=tf_efficientdet_d7 --b=1 --checkpoint=d7.pth
在GPU上batch_size=1，推理速度
Test: [4270/5000]  Time: 0.964s (0.950s,    1.05/s)
Test: [4280/5000]  Time: 0.911s (0.950s,    1.05/s)
Test: [4290/5000]  Time: 0.977s (0.950s,    1.05/s)
Test: [4300/5000]  Time: 0.960s (0.950s,    1.05/s)
Test: [4310/5000]  Time: 0.939s (0.950s,    1.05/s)
Test: [4320/5000]  Time: 0.957s (0.950s,    1.05/s)
Test: [4330/5000]  Time: 0.951s (0.950s,    1.05/s)
Test: [4340/5000]  Time: 0.929s (0.950s,    1.05/s)
Test: [4350/5000]  Time: 0.956s (0.950s,    1.05/s)
Test: [4360/5000]  Time: 0.974s (0.950s,    1.05/s)
Test: [4370/5000]  Time: 0.955s (0.950s,    1.05/s)
Test: [4380/5000]  Time: 0.963s (0.950s,    1.05/s)
Test: [4390/5000]  Time: 0.949s (0.950s,    1.05/s)
Test: [4400/5000]  Time: 0.950s (0.950s,    1.05/s)
Test: [4410/5000]  Time: 0.924s (0.950s,    1.05/s)
Test: [4420/5000]  Time: 1.000s (0.950s,    1.05/s)
Test: [4430/5000]  Time: 0.977s (0.950s,    1.05/s)
Test: [4440/5000]  Time: 0.923s (0.950s,    1.05/s)
Test: [4450/5000]  Time: 0.918s (0.950s,    1.05/s)
Test: [4460/5000]  Time: 0.946s (0.950s,    1.05/s)
Test: [4470/5000]  Time: 0.958s (0.950s,    1.05/s)
Test: [4480/5000]  Time: 0.978s (0.950s,    1.05/s)
Test: [4490/5000]  Time: 0.976s (0.950s,    1.05/s)
对比：1.50*4>1.05
```