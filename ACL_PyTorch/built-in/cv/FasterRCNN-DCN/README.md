# FasterRCNN-DCN模型PyTorch离线推理指导

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

3. 获取权重文件  

从https://download.openmmlab.com/mmdetection/v2.0/dcn/下载对应的权重文件

4. 数据集    
   本模型使用coco2017的验证集验证 

5. 因包含dcn自定义算子，去除对onnx的检查  
   将/usr/local/python3.7.5/lib/python3.7/site-packages/torch/onnx/utils.py的_check_onnx_proto(proto)改为pass  

6. 使用mmdet框架自带的脚本导出onnx即可，这里指定shape为1216。由于当前框架限制，仅支持batchsize=1的场景

   ```
   python3.7 mmdetection/tools/pytorch2onnx.py mmdetection/configs/dcn/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py ./faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco_20200203-4f85c69c.pth --output-file=FasterRCNNDCN.onnx --shape=1216
   ```

7. 运行atc.sh脚本，完成onnx到om模型的转换，注意输出节点可能需要根据实际的onnx修改

   ```
   bash atc.sh FasterRCNNDCN.onnx FasterRCNNDCN
   ```

   

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  

1. 数据预处理

   ```
   python3.7 mmdetection_coco_preprocess.py --image_folder_path ./val2017 --bin_folder_path val2017_bin
   python3.7 get_info.py bin ./val2017_bin coco2017.info 1216 1216
   ```

2. 使用benchmark工具进行推理

```
./benchmark.x86_64 -model_type=vision -batch_size=1 -device_id=0 -input_text_path=./coco2017.info -input_width=1216 -input_height=1216 -useDvpp=False -output_binary=true -om_path=FasterRCNNDCN.om
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

|       模型        | 官网pth精度 | 310离线推理精度 | gpu性能 | 310性能  |
| :---------------: | :---------: | :-------------: | :-----: | :------: |
| FasterRCNNDCN bs1 |  map:0.445  |    map:0.444    |         | 0.357fps |



