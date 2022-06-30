# CasCadeRCNN-DCN模型PyTorch离线推理指导

## 1 环境准备 

1. 安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip3.7 install -r requirements.txt  
```

2. 获取，修改与安装开源模型代码  

```
git clone https://github.com/open-mmlab/mmdetection.git   
cd mmdetection  
git reset a21eb25535f31634cef332b09fc27d28956fb24b --hard
pip3.7 install -v -e .
cd ..
```

将提供的pytorch_code_change文件夹中的文件替换原文件

```
cp ./pytorch_code_change/bbox_nms.py ./mmdetection/mmdet/core//post_processing/bbox_nms.py
cp ./pytorch_code_change/rpn_head.py ./mmdetection/mmdet/models/dense_heads/rpn_head.py
cp ./pytorch_code_change/single_level_roi_extractor.py ./mmdetection/mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py
cp ./pytorch_code_change/delta_xywh_bbox_coder.py ./mmdetection/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py
cp ./pytorch_code_change/deform_conv.py /root/anaconda3/envs/dcn/lib/python3.7/site-packages/mmcv/ops/deform_conv.py
#以上为DCNV1版本，若需要使用DCNV2,则需要
cp ./pytorch_code_change/modulated_deform_conv.py /root/anaconda3/envs/dcn/lib/python3.7/site-packages/mmcv/ops/modulated_deform_conv.py
```

3. 获取权重文件  

从https://github.com/open-mmlab/mmdetection/blob/master/configs/dcn/README.md下载对应的权重文件，对应R-101-FPN Cascade行对应的权重，点击model下载即可

4. 数据集    
   本模型使用coco2017的验证集验证 

5. 因包含dcn自定义算子，去除对onnx的检查  
   将/usr/local/python3.7.5/lib/python3.7/site-packages/torch/onnx/utils.py的_check_onnx_proto(proto)改为pass  

6. 使用mmdet框架自带的脚本导出onnx即可，这里指定shape为1216。由于当前框架限制，仅支持batchsize=1的场景

   ```
   python3.7.5 mmdetection/tools/pytorch2onnx.py mmdetection/configs/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py ./cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-3b2f0594.pth --output-file=cascadeRCNNDCN.onnx --shape=1216
   ```

7. 运行atc.sh脚本，完成onnx到om模型的转换，注意输出节点可能需要根据实际的onnx修改，若设备为310，则需要修改atc.sh脚本中的--soc_version为Ascend310

   ${chip_name}可通过`npu-smi info`指令查看
   
    ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

   ```
   bash atc.sh cascadeRCNNDCN.onnx cascadeRCNNDCN Ascend${chip_name} # Ascend310P3
   ```
   

   

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  

1. 数据预处理

   ```
   python3.7 mmdetection_coco_preprocess.py --image_folder_path ./val2017 --bin_folder_path val2017_bin
   python3.7 get_info.py bin ./val2017_bin coco2017.info 1216 1216
   python3.7 get_info.py jpg ./val2017 coco2017_jpg.info
   ```

2. 使用benchmark工具进行推理

```
./benchmark.x86_64 -model_type=vision -batch_size=1 -device_id=0 -input_text_path=./coco2017.info -input_width=1216 -input_height=1216 -useDvpp=False -output_binary=true -om_path=cascadeRCNNDCN.om
```

3. 推理结果展示

本模型提供后处理脚本，将二进制数据转化为txt文件，同时生成画出检测框后的图片。执行脚本

```
python3.7 mmdetection_coco_postprocess.py --bin_data_path=result/dumpOutput_device0 --prob_thres=0.05 --ifShowDetObj --det_results_path=detection-results --test_annotation=coco2017_jpg.info
```

4. 精度验证

   ```
   python3.7 txt_to_json.py
   python3.7 coco_eval.py
   ```

   

**评测结果：**   

|        模型         | 官网pth精度 | 310离线推理精度 | gpu性能 |     310性能/310P性能     |
| :-----------------: | :---------: | :-------------: | :-----: | :---------------------: |
| CascadedRCNNDCN bs1 |  map:0.45   |    map:0.45     | 4.6fps  | 1.9258fps/fps/2.9534fps |



