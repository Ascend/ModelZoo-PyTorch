# FasterRCNN-FPN-DCN模型PyTorch离线推理指导(NPU:310P、CANN:5.1RC1)



## 1 环境准备

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

```
pip3.7 install -r requirements.txt  
```  
   说明：PyTorch选用开源1.8.0版本    

2.获取，修改与安装开源模型代码（安装mmcv与mmdetection）

注：
1:python3.7 setup.py develop执行较慢，耐心等候。2:安装在FasterRCNN-FPN-DCN文件夹下

```
git clone https://github.com/open-mmlab/mmcv -b master 
cd mmcv
git checkout v1.2.7
MMCV_WITH_OPS=1 pip3.7 install -e .
patch -p1 < ../mmcv.patch
cd ..
git clone https://github.com/open-mmlab/mmdetection -b master
cd mmdetection
git reset --hard a21eb25535f31634cef332b09fc27d28956fb24b
patch -p1 < ../dcn.patch
pip3.7 install -r requirements/build.txt
python3.7 setup.py develop
cd ..
```
3.获取权重文件

``` 
cd mmdetection 
mkdir checkpoints
cd checkpoints
``` 

[faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth](参照指导书文档)

4.数据集(用户自行准备好数据集，本文的以coco验证集为例)

[测试集]coco_val2017

[标签]instances_val2017.json

存放路径说明：val2017存放5000张验证集图片，annotations存放instances_val2017.json文件
```
FasterRCNN-FPN-DCN
|——data
| |——coco
| | |——val2017
| | |——annotations
```
5.[获取benchmark工具](参照指导书文档)  
  将benchmark.x86_64或benchmark.aarch64放到当前目录  
  
  
## 2 离线推理

310P上执行，执行时使npu-smi info查看设备状态，确保device空闲（本模型推理时显存占用较大，需设备空闲时才能测试出正常性能指标）

获取faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth权重文件，放在1.3创建的目录下。依次执行下面两条sh命令可以完成整个推理流程。

注1：详细的步骤细节，请参考推理指导书的对应内容。

注2：该sh脚本里生成om模型的ATC语句调用的NPU设备的代码为：

--soc_version=Ascend${chip_name}

${chip_name}为用户当前设备的NPU型号，需要用户自己设置。可以通过"npu-smi info”查看。
```
# 1：pth转om模型
bash pth2om.sh
# 得到onnx以及om模型。

# 2：执行推理和评估脚本
bash eval_acc.sh
# 此步骤包含：数据预处理、模型推理（输出性能数据）、数据后处理、输出精度数据。
```

**评测结果：**


|模型|batch_size|官网pth精度|T4基准性能|310理线推理精度|310性能|310P离线推理精度|310P性能|
|---|---|---|---|---|---|---|---|
|faster_rcnn_r50_fpn_dcn|1|[box AP:41.3%](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn)|5.40FPS|box AP:41.2%|4.61FPS|box AP:41.1%|7.41FPS|
|faster_rcnn_r50_fpn_dcn|4|-|4.00FPS|-|6.68FPS|-|8.81FPS|
|faster_rcnn_r50_fpn_dcn|8|-|3.60FPS|-|7.21FPS|-|8.45FPS|
|faster_rcnn_r50_fpn_dcn|16|-|显存不够|-|显存不够|-|8.71FPS|

最优batch性能对比：

310P/t4 : 8.81/5.40  ≈ 1.63 > 1.6

310P/310: 8.81/7.21  ≈ 1.22 > 1.2

精度对比：（注：本模型为静态模型，因网络和后处理代码的影响。只能得到bs=1下精度数据）

0.412/0.413 ≈ 99.76% ＞ 99%

0.411/0.413 ≈ 99.52% ＞ 99%