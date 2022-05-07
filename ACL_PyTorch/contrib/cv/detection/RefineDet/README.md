# RefineDet模型PyTorch离线推理指导

## Ascend310：
 

## 1 环境准备

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

```
pip3.7 install -r requirements.txt
```

2.获取代码和[权重文件](https://drive.google.com/file/d/1RCCTaNeby0g-TFE1Cvjm3dYweBiyyPoq/view?usp=sharing)，放到当前路径下

```
git clone https://github.com/luuuyi/RefineDet.PyTorch.git -b master
cd RefineDet.PyTorch
git reset --hard 0e4b24ce07245fcb8c48292326a731729cc5746a
patch -p1 < ../refinedet.patch

```

3.获取数据集,[VOC数据集](http://host.robots.ox.ac.uk/pascal/VOC)，可以通过下面的命令下载


```
sh data/scripts/VOC2007.sh
cd ../
```
4.获取[benchamrk](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)

把benchmark.x86_64 或者 benchmark.aarch64 放到当前目录


## 2 离线推理

pth转换为om
```
bash test/pth2om.sh 
```


精度,性能测试

```
 bash test/eval_acc_perf.sh --datasets_path=/root/datasets/VOCdevkit/
```




**评测结果：**

| 模型      | pth精度  | 310精度  |    基准性能    |310性能  |
| :------: | :------: | :------: | :------:  | :------:  | 
| RefineDet bs1  | [mAP:79.81%](https://github.com/luuuyi/RefineDet.PyTorch) | mAP:79.56%|  63.94fps | 101.24fps |
| RefineDet bs16 | [mAP:79.81%](https://github.com/luuuyi/RefineDet.PyTorch) |mAP:79.56% |  72.77fps | 136.8fps |




备注：

- nms放在后处理，在cpu上计算
- onnx转om时，不能使用fp16，否则精度不达标
  ```
  --precision_mode allow_fp32_to_fp16
  ```


## Ascend710：

##  文件说明

```  
├── test                          //打印精度性能脚本文件夹
├── eval_utils.py                 //后处理精度计算时需要用到的函数  
├── benchmark.aarch64            //离线推理工具（适用ARM架构）  
├── benchmark.x86_64             //离线推理工具（适用x86架构）  
├── get_prior_data.py             //获取prior_data的脚本  
├── get_info.py                   //用于获取二进制数据集信息的脚本    
├── RefineDet320_VOC_final.pth       //训练后的权重文件  
├── RefineDet320_VOC_final_no_nms.onnx     //onnx文件
├── RefineDet_postprocess.py      //后处理精度计算脚本  
├── RefineDet_preprocess.py     //数据集预处理脚本 
├── RefineDet_pth2onnx.py        //pth转onnx脚本  
├── refinedet_voc_320_non_nms_bs1_710.om   //推理om文件 
├── refinedet_voc_320_non_nms_bs32_710.om //推理om文件  
├── refinedet.patch             //开源代码补丁  
├── requirements.txt            //环境依赖
├── LICENSE
├── modelzoo_level.txt
└── README.md 
```

## 1 环境准备

1. 安装必要的依赖

```
pip3.7 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. 获取开源代码仓放到当前路径下(https://github.com/luuuyi/RefineDet.PyTorch)

```
git clone https://github.com/luuuyi/RefineDet.PyTorch.git
cd RefineDet.PyTorch
git checkout master 
git reset --hard   0e4b24ce07245fcb8c48292326a731729cc5746a
```

3. 执行“refinedet.patch”文件

```
patch -p1 <  ../refinedet.patch
cd ..
```

## 2 准备数据集

1. 获取原始数据集

本模型支持VOC2007 4952张图片的验证集。请用户需自行获取VOC2007数据集，上传数据集到服务器任意目录并解压（如：/root/datasets/VOCdevkit/）。

文件目录结构如下：

```
VOC2007
├──SegmentationObject
├──SegmentationClass
├──JPEGImages
├──Annotations
├──ImageSets
│   ├── Segmentation
│   ├── Main
│   ├── Layout
```

2. 数据预处理

运行“RefineDet_preprocess.py”脚本将jpg文件转化为二进制bin文件。

```
python3.7 RefineDet_preprocess.py '/root/datasets/VOCdevkit/' voc07test_bin
```

- “/root/datasets/VOCdevkit/”：数据集的路径。

- “voc07test_bin”：生成的bin文件路径。

3. 生成数据集info文件

执行“get_info.py”脚本，输入已经获得的图片文件，输出生成图片数据集的info文件。

```
python3.7 get_info.py voc07test_bin voc07test.info
```
- “voc07test_bin”：预处理数据集路径。

- “voc07test.info”：生成的info文件的相对路径。

运行结束后生成“voc07test.info”。

## 3 模型推理

1. 模型转换

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

a.获取权重文件。 

从源码包中获取权重文件“RefineDet320_VOC_final.pth”。

b.  导出onnx文件。

执行“RefineDet_pth2onnx.py”脚本导出onnx文件。

```
python3.7 RefineDet_pth2onnx.py './RefineDet320_VOC_final.pth'  'RefineDet320_VOC_final_no_nms.onnx' '/root/datasets/VOCdevkit/'
```

- “./RefineDet320_VOC_final.pth”：权重文件路径。

- “RefineDet320_VOC_final_no_nms.onnx”：生成的onnx文件。

- “/root/datasets/VOCdevkit/”：数据集路径。

运行结束得到 “RefineDet320_VOC_final_no_nms.onnx”文件。

2. 使用ATC工具将ONNX模型转OM模型

a.  设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
（该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。)
```

b.  执行命令得到om模型

```
atc --framework=5 --out_nodes="Reshape_239:0;Softmax_246:0;Reshape_226:0;Softmax_233:0" --model=RefineDet320_VOC_final_no_nms.onnx 
--output=refinedet_voc_320_non_nms_bs1_710 --input_format=NCHW --input_shape="image:1,3,320,320" --log=debug 
--soc_version=Ascend710 --precision_mode allow_fp32_to_fp16 
（切换不同batchsize时需要修改参数--output中的文件名"_nms_bs<batchsize>_710"和--input_shape="<batchsize>,3,320,320"）
（默认使用的是Ascend710处理器，当使用Ascend310处理器时，应当修改对应参数"--soc_version=Ascend310"和"--output"后缀名为310）
```

- 参数说明
- --model：为ONNX模型文件。
- --framework：5代表ONNX模型。
- --output：输出的OM模型。
- --input_format：输入数据的格式。
- --input_shape：输入数据的shape。
- --log：日志级别。
- --soc_version：处理器型号。

```
若atc执行出错，错误代码为E10016，请使用Netron工具查看对应Reshape节点和Softmax节点，并修改代码。
```

3. 开始推理验证

a.  使用Benchmark工具进行推理

执行以下命令增加Benchmark工具可执行权限，并根据OS架构选择工具，如果是X86架构，工具选择benchmark.x86_64，如果是Arm，选择benchmark.aarch64 。

```
chmod u+x benchmark.${arch}
```

benchmark.${arch}：选择对应操作系统的benchmark工具，如benchmark.x86_64或benchmark.aarch64 。

b.  执行推理

```
./benchmark.x86_64  -model_type=vision  -device_id=0  -batch_size=1  -om_path=./refinedet_voc_320_non_nms_bs1_710.om  
 -input_text_path=./voc07test.info  -input_width=320 -input_height=320  -output_binary=True  -useDvpp=False
 (切换不同batchsize时需要修改参数 -batch_size=<batchsize>和对应的om文件名即参数 -om_path)
 (默认推理的是Ascend710处理器下转换的om文件，如要推理Ascend310处理器下转换的om文件，应当修改对应参数"--om_path="的后缀为310)
```

推理后的输出默认在当前目录result下。

- 参数说明
- -model_type：模型类型
- -om_path：om文件路径
- -device_id：NPU设备编号
- -batch_size：参数规模
- -input_text_path：图片二进制信息
- -input_width：输入图片宽度
- -input_height：输入图片高度
- -useDvpp：是否使用Dvpp
- -output_binary：输出二进制形式

性能相关文件也输出在目录result下

c.  精度验证

调用“RefineDet_postprocess .py”脚本，可以获得Accuracy数据，结果保存在result.json中。

```
python3.7 get_prior_data.py
```
```
mv result/dumpOutput_device0 result/dumpOutput_device0_bs1
(不同batchsize文件名需修改"_bs<batchsize>")
```
```
python3.7 RefineDet_postprocess.py --datasets_path '/root/datasets/VOCdevkit/' --result_path result/dumpOutput_device0_bs1   > result_bs1.json 
(不同batchsize文件名需修改"_bs<batchsize>",即修改参数--result_path，还需修改输出的json文件名"_bs<batchsize>")
```

- 参数说明
- --datasets_path：数据集路径。
- --result_path：推理得到的bin文件路径。

最后推理精度结果储存在"result_bs1.json"中

## 精度和性能

**评测结果：**

|       模型       | pth精度  | 310精度  |310性能  |710精度  |710性能
|:--------------:| :------: | :------: | :------:  | :------:  |  :------:  |
| RefineDet bs1  | [mAP:79.81%](https://github.com/luuuyi/RefineDet.PyTorch) | mAP:79.56%|664.24fps|mAP:79.58%|917.81fps
| RefineDet bs32 | [mAP:79.81%](https://github.com/luuuyi/RefineDet.PyTorch) |mAP:79.56% | 928.72fps|mAP:79.58%|1454.03fps

备注：

- nms放在后处理，在cpu上计算
- onnx转om时，不能使用fp16，否则精度不达标
- 310性能数据需要乘以4

```
--precision_mode allow_fp32_to_fp16
```

