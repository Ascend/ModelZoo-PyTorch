# FasterRCNN-FPN-DCN模型PyTorch离线推理指导(NPU:710)

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

710上执行，执行时使npu-smi info查看设备状态，确保device空闲（本模型推理时显存占用较大，需设备空闲时才能测试出正常性能指标）




```
# 1：生成onnx模型
source pth2onnx.sh

# 2：修改onnx模型（pth转出的onnx模型在转om时会报部分结点参数类型不一致的错误）
source correctonnx.sh

# 3:生成om模型
source onnx2om.sh

# 4:生成coco数据集相关文件
python3.7 FasterRCNN+FPN+DCN_preprocess.py --image_folder_path ./data/coco/val2017 --bin_folder_path coco2017_bin	 
python3.7 gen_dataset_info.py bin coco2017_bin coco2017_bin.info 1216 1216
python3.7 gen_dataset_info.py jpg ./data/coco/val2017 coco2017_jpg.info

# 5:om模型推理
source inf.sh

# 6:数据后处理
python3.7 FasterRCNN+FPN+DCN_postprocess.py --test_annotation coco2017_jpg.info --bin_data_path result/dumpOutput_device0

# 7：coco eval验证，获取精度数据
python3.7 txt2json.py --npu_txt_path detection-results --json_output_file coco_detection_result
python3.7 coco_eval.py --ground_truth data/coco/annotations/instances_val2017.json --detection_result coco_detection_result.json

# 8:纯推理测试性能
source framerate.sh
```

**评测结果：**


|模型|batch_size|官网pth精度|T4基准性能|310理线推理精度|310性能|710离线推理精度|710性能|
|---|---|---|---|---|---|---|---|
|faster_rcnn_r50_fpn_dcn|1|[box AP:41.3%](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn)|5.40FPS|box AP:41.2%|4.61FPS|box AP:41.1%|7.41FPS|
|faster_rcnn_r50_fpn_dcn|4|-|4.00FPS|-|6.68FPS|-|8.81FPS|
|faster_rcnn_r50_fpn_dcn|8|-|3.60FPS|-|7.21FPS|-|8.45FPS|
|faster_rcnn_r50_fpn_dcn|16|-|显存不够|-|显存不够|-|8.71FPS|

