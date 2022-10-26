**DE⫶TR**: End-to-End Object Detection with Transformers 

### 推理流程
#### 环境准备
pip3 install -r requirement.txt
```
克隆本代码仓后
cd Detr
git clone 参考指导书链接
cd detr
git checkout b9048ebe86561594f1472139ec42327f00aba699
patch -p1 < ../detr.patch

detr.pth权重文件获取
链接: https://pan.baidu.com/s/1iz18BwU6E141hEmwigpe_w  密码: du65
下载完成后
mkdir model
mv detr.pth ./model
coco数据集下载地址：参考指导书链接
标注json文件下载地址：参考指导书链接
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
* 1、运行pth2onnx.py
```
batch_size=1
python3.7 pth2onnx.py --batch_size=1 
batch_size=4
python3.7 pth2onnx.py --batch_size=4 
```
batch_size:根据需求选择batch大小

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
```
* 运行onnx2om.py

```
mkdir auto_om
batch_size=1
python3.7 onnx2om.py --batch_size=1 
batch_size=4
python3.7 onnx2om.py --batch_size=4 
```
#### 测试集预处理
coco val 5k 数据集下载
```
cd Detr
python3.7 preprocess.py --datasets=coco_data/val2017 --img_file=img_file --mask_file=mask_file
```
选择coco val数据集进行验证  
datasets:coco 验证集的路径   
img_file：生成图片 bin文件路径  
mask_file:生成mask bin文件路径

#### 离线推理

使用msame工具进行离线推理
* 1、下载tools仓
```
git clone https://gitee.com/ascend/tools.git
cd tools/msame
设置环境变量
chmod +x build.sh
./build.sh g++ $HOME/AscendProjects/tools/msame/out
```
* 2、执行msame推理
```
batch_size=1
python3.7 excute_omval.py --img_path=img_file --mask_path=mask_file --out_put=out_put --result=result --batch_size=1 > bs1_time.log
batch_size=4
python3.7 excute_omval.py --img_path=img_file --mask_path=mask_file --out_put=out_put --result=result --batch_size=4 > bs4_time.log 
```
执行该脚本，推理结果路径最终在result目录下，并生成推理info日志文件
img_path:前处理的图片文件路径  
mask_path:前处理的mask文件路径  
out_put:msame推理数据输出路径  
result:推理数据最终汇总路径  
batch_size:batch大小，可选1或4  


#### 测试数据后处理
```
python3.7 postprocess.py --coco_path=coco_data/val2017 --result=result
```
参数解释：  
coco_path：测试集目录   
result:om推理出的数据存放路径  

## Detr测试结果:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.416
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.620
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.440
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.191
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.455
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.614
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.330
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.528
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.569
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.305
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.625
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.805
```
map=41.6>42.0*0.99  
精度达标
#### 性能测试
```
batch_size=1
python3.7 FPS.py --log_path=bs1_time.log
FPS=21.9
batch_size=4
python3.7 FPS.py --log_path=bs4_time.log
FPS=28.1
解析推理日志文件，计算性能数据
log_path：推理info日志文件

GPU在线推理
cd detr 
python3.7 main.py --batch_size 1 --no_aux_loss --eval --resume ../model/detr.pth --coco_path /path/to/coco
在GPU上batch_size=1，推理速度
FPS=10.9
在GPU上batch_size=4，推理速度
FPS=10.9

21.9>10.9
28.1>10.9
性能达标
```