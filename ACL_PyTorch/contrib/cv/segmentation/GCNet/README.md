# GCNet模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip3.7 install -r requirements.txt  
```



2.获取，修改与安装开源模型代码  

```
git clone https://github.com/open-mmlab/mmcv
cd mmcv
git reset --hard 643009e4458109cb88ba5e669eec61a5e54c83be
pip install -e .
cd ..
git clone https://github.com/open-mmlab/mmdetection
cd mmdetection
git reset --hard 6c1347d7c0fa220a7be99cb19d1a9e8b6cbf7544
pip install -r requirements/build.txt
python setup.py develop
patch -p1 < GCNet.diff
```



3.获取权重文件  

从[LINK](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet)中获取权重文件，将权重文件mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco_20200204-17235656.pth放到当前工作目录 （执行pth2onnx时会自动下载）

4.数据集     

使用COCO官网的coco2017的5千张验证集进行测试，请参考原始开源代码仓mmdetection中对公共数据集的设置

5.[获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)  
将benchmark.x86_64或benchmark.aarch64放到当前工作目录  



## 2 模型转换

1.pth转onnx模型

```
python tools/deployment/pytorch2onnx.py configs/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco.py mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco_20200204-17235656.pth --output-file  GCNet.onnx --input-img demo/demo.jpg --test-img tests/data/color.jpg --shape 800 1216
```



2.onnx转om模型

```
atc --framework=5 --model=GCNet.onnx --output=./GCNet_bs1 --input_shape="input:1,3,800,1216"  --log=error --soc_version=Ascend310
```



3.执行以下命令生成om模型文件

```
bash test/pth2om.sh
```



## 3 离线推理

1.数据预处理

```
python GCNet_preprocess.py --image_src_path=${datasets_path}/val2017 --bin_file_path=val2017_bin --model_input_height=800 --model_input_width=1216 
```



2.生成数据集信息文件

```
python gen_dataset_info.py bin val2017_bin coco2017.info 1216 800
python gen_dataset_info.py jpg val2017 coco2017_jpg.info
```



3.执行离线推理

```
./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=1 -om_path=./GCNet_bs1.om -input_text_path=./coco2017.info  -input_width=1216 -input_height=800 -output_binary=True -useDvpp=False
```



4.使用后处理脚本展示推理结果

```
python GCNet_postprocess.py --bin_data_path=./result/dumpOutput_device1/ --test_annotation=coco2017_jpg.info --det_results_path=detection-results --annotations_path=annotations/instances_val2017.json --net_out_num=3 --net_input_height=800 --net_input_width=1216
```



5.NPU精度测试

```
python txt_to_json.py
python coco_eval.py 
```



6.NPU性能测试

```
./benchmark.x86_64 -round=20 -om_path=GCNet_bs1.om -device_id=1 -batch_size=1
```



7.GPU性能测试

onnx包含自定义算子，因此不能使用开源TensorRT测试性能数据，故在T4机器上使用pth在线推理测试性能数据

测评T4精度与性能

```
python tools/test.py configs/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco.py ./mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco_20200204-17235656.pth --eval bbox
python coco_eval.py
```



8.执行命令进行离线推理

```
bash test/eval_acc_perf.sh
```



 **评测结果：**   

|   模型    | 官网pth精度 | 310离线推理精度 | 基准性能 | 310性能  |
| :-------: | :---------: | :-------------: | :------: | :------: |
| GCNet bs1 |  mAP:0.613  |    mAP:0.611    | 3.931fps | 8.144fps |

备注：  
1.GCNet的mmdetection实现不支持多batch。

2.onnx包含自定义算子，因此不能使用开源TensorRT测试性能数据，故在T4机器上使用pth在线推理测试性能数据。

说明：

1.om推理box map50精度为0.611，T4推理box map50精度为0.613，精度下降在1个点之内，因此可视为精度达标。

2.batch1：2.036 * 4 fps > 3.931fps 即310单个device的吞吐率乘4即单卡吞吐率约为T4单卡的吞吐率2倍，故310性能高于T4性能，性能达标。

