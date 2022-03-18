# ErfNet模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```

2.获取，修改与安装开源模型代码  
```
git clone https://github.com/Eromera/erfnet_pytorch   
cd erfnet_pytorch  
git reset d4a46faf9e465286c89ebd9c44bc929b2d213fb3 --hard
cd ..  
``` 

3.获取权重文件  
[erfnet_pretrained.pth](https://github.com/Eromera/erfnet_pytorch/blob/master/trained_models/erfnet_pretrained.pth)   

4.数据集     
[获取cityscapes](https://www.cityscapes-dataset.com/)
- Download the Cityscapes dataset from https://www.cityscapes-dataset.com/

  - Download the "leftImg8bit" for the RGB images and the "gtFine" for the labels.
  - Please note that for training you should use the "_labelTrainIds" and not the "_labelIds", you can download the [cityscapes scripts](https://github.com/mcordts/cityscapesScripts) and use the [conversor](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py) to generate trainIds from labelIds  

5.[获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)  
将benchmark.x86_64放到当前目录  

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  
```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/root/datasets  
```
 **评测结果：**   
| 模型      | 官网pth精度  | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| ErfNet bs1  | [iou:72.20](https://github.com/Eromera/erfnet_pytorch) | iou:72.19 |  47.59fps | 214.3452fps | 
| ErfNet bs16 | [iou:72.20](https://github.com/Eromera/erfnet_pytorch) | iou:72.19 | 63.34fps | 175.6904fps | 

备注：  
1.由于使用原始的onnx模型转出om后，精度有损失，故添加了modify_bn_weights.py来修改转出onnx模型bn层的权重。
2.由于tensorRT不支持部分算子，故gpu性能数据使用在线推理的数据。


