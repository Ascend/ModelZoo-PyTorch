# FasterRCNN-FPN-DCN模型PyTorch离线推理指导

## 1 环境准备

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

```
pip3.7 install -r requirements.txt  
```  
   说明：PyTorch选用开源1.8.0版本    
2.获取，修改与安装开源模型代码

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
```
3.获取权重文件

``` 
cd mmdetection 
mkdir checkpoints
cd checkpoints

``` 

[faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth](参照指导书文档)
4.数据集  

[测试集]参照指导书文档  
[标签]参照指导书文档  

5.[获取benchmark工具](参照指导书文档)  
  将benchmark.x86_64或benchmark.aarch64放到当前目录  
  
  
## 2 离线推理

310上执行，执行时使npu-smi info查看设备状态，确保device空闲

```
#OM model generation
bash test/pth2onnx.sh
bash test/onnx2om.sh

#COCO dataset preprocess
python3.7 FasterRCNN+FPN+DCN_preprocess.py --image_folder_path coco/val2017 --bin_folder_path coco2017_bin	 
python3.7 gen_dataset_info.py bin coco2017_bin coco2017_bin.info 1216 1216
python3.7 gen_dataset_info.py jpg coco/val2017 coco2017_jpg.info

#OM model inference
bash test/inf.sh

#Inference result postprocess
python3.7 FasterRCNN+FPN+DCN_postprocess.py --test_annotation coco2017_jpg.info --bin_data_path result/dumpOutput_device0

#COCO eval
python3.7 txt2json.py --npu_txt_path detection-results --json_output_file coco_detection_result
python3.7 coco_eval.py --groud_truth coco/annotations/instances_val2017.json --detection_result coco_detection_result.json

#FrameRate eval
bash test/framerate.sh
```

**评测结果：**

| 模型 | 官网pth精度                                                  | 310离线推理精度 | 基准性能 | 310性能 |
| ---- | ------------------------------------------------------------ | --------------- | -------- | ------- |
| faster_rcnn_r50_fpn_dcn | [box AP:41.3%](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn) | box AP:41.2%    | 5.2fps   | 2.8fps |

