# Cascade-RCNN-Resnet101-FPN-DCN模型PyTorch离线推理指导

## 1 环境准备

1. 安装必要的依赖

   在文件夹中处理。
   
   测试环境可能已经安装其中的一些不同版本的库了，故手动测试时建议创建虚拟环境后自己安装，参考开源仓代码的获取方式：

```
conda create -n dcn python=3.7
conda activate dcn
pip install onnx==1.7.0
pip install onnxruntime==1.9.0
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cpuonly -c pytorch
pip install mmcv-full==1.2.4
```

2. 获取，修改与安装开源模型代码，参考开源仓代码的获取方式：

```
git clone https://github.com/open-mmlab/mmdetection.git   
cd mmdetection  
git reset a21eb25535f31634cef332b09fc27d28956fb24b --hard
pip install -v -e .
cd ..
```

将提供的**pytorch_code_change**文件夹中的文件替换原文件。

```
cp ./pytorch_code_change/bbox_nms.py ./mmdetection/mmdet/core//post_processing/bbox_nms.py
cp ./pytorch_code_change/rpn_head.py ./mmdetection/mmdet/models/dense_heads/rpn_head.py
cp ./pytorch_code_change/single_level_roi_extractor.py ./mmdetection/mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py
cp ./pytorch_code_change/delta_xywh_bbox_coder.py ./mmdetection/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py
cp ./pytorch_code_change/pytorch2onnx.py ./mmdetection/tools/pytorch2onnx.py
cp ./pytorch_code_change/cascade_rcnn_r50_fpn.py ./mmdetection/configs/_base_/models/cascade_rcnn_r50_fpn.py
cp ./pytorch_code_change/deform_conv.py /root/anaconda3/envs/dcn/lib/python3.7/site-packages/mmcv/ops/deform_conv.py
#注意这里要根据实际情况下的安装路径来修改
```

3. 获取权重文件

参考源码仓的方式获取，可以通过obs方法获取，下载对应的权重文件。

```
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/detection/Cascade%20RCNN-Resnet101-FPN-DCN/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-3b2f0594.pth
```

4. 数据集 
   本模型使用coco2017的验证集（val2017）验证，将服务器上的数据集复制到本文件下固定位置:data/coco/annotation/instances_val2017.json以及data/coco/val2017

   ```
   mkdir -p data/coco
   cp -r /opt/npu/datasets/coco/* /home/tyjf/data/coco

   ```
   
5. 导出onnx

   使用mmdet框架自带的脚本导出onnx即可，这里指定shape为1216。

   由于当前框架限制，仅支持batchsize=1的场景。

```
python mmdetection/tools/pytorch2onnx.py mmdetection/configs/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py ./cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-3b2f0594.pth --output-file=cascadeR101dcn.onnx --shape=1216 --verify --show
```

6. 导出om

   运行atc.sh脚本，完成onnx到om模型的转换，注意输出节点可能需要根据实际的onnx修改。
   
```
bash atc.sh cascadeR101dcn.onnx cascadeR101dcn
```


## 2 离线推理 

在310上执行，执行时使npu-smi info查看设备状态，确保device空闲。

1. 数据预处理

```
python mmdetection_coco_preprocess.py --image_folder_path ./data/coco/val2017 --bin_folder_path val2017_bin
python get_info.py bin ./val2017_bin coco2017.info 1216 1216
python get_info.py jpg ./data/coco/val2017 coco2017_jpg.info
```

2. 使用benchmark工具进行推理

```
chmod u+x benchmark.x86_64
./benchmark.x86_64 -model_type=vision -batch_size=1 -device_id=0 -input_text_path=./coco2017.info -input_width=1216 -input_height=1216 -useDvpp=False -output_binary=true -om_path=cascadeR101dcn.om
```

3. 推理结果展示

本模型提供后处理脚本，将二进制数据转化为txt文件，同时生成画出检测框后的图片。执行脚本

```
python mmdetection_coco_postprocess.py --bin_data_path=result/dumpOutput_device0 --prob_thres=0.05 --ifShowDetObj --det_results_path=detection-results --test_annotation=coco2017_jpg.info
```

4. 精度验证

```
python txt_to_json.py
python coco_eval.py --ground_truth ./data/coco/annotation/instances_val2017.json
```
可以看到NPU精度：'bbox_mAP': 0.452

5. 性能验证

查看NPU性能

```
bash test/perf_npu.sh
#或者运行 ./benchmark.x86_64 -round=50 -om_path=cascadeR101dcn.om --device_id=2 -batch_size=1
```

可以看到NPU性能：

[INFO] ave_throughputRate: 0.620627samples/s, ave_latency: 1593.71ms

0.65281*4=2.61fps

6. GPU性能与精度验证

由于模型算子的原因采取在线推理的方式检测GPU性能：

在GPU上搭建好环境，并进行预处理：

```
mkdir -p data/coco/val2017
cp -r /root/coco/val2017/* /home/dcnv0/data/coco/val2017
mkdir -p data/coco/annotations
cp -r /root/coco/annotations/* /home/dcnv0/data/coco/annotations
```

测试性能与精度：

```
cd /home/dcnv0
conda activate cascade
python ./mmdetection/tools/test.py ./mmdetection/configs/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py ./cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-3b2f0594.pth --eval=bbox
```

可以算出GPU性能为4.06fps左右。

**评测结果：**   

|                模型                | 官网pth精度 | 310离线推理精度 | gpu性能 | 310性能 |
| :--------------------------------: | :---------: | :-------------: | :-----: | :-----: |
| Cascade-RCNN-Resnet101-FPN-DCN |  mAP:0.45   |    mAP:0.452    | 4.06fps | 2.64fps |
