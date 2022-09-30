文件作用说明：

1.  env.sh：ATC工具环境变量配置脚本
2.  require.txt：脚本运行所需的第三方库
3.  parse_json.py： coco数据集标签json文件解析脚本 
4.  preprocess_yolov3_pytorch.py： 二进制数据集预处理脚本
5.  get_coco_info.py： yolov3.info生成脚本 
6.  bin_to_predict_yolo_pytorch.py： benchmark输出bin文件解析脚本
7.  map_calculate.py： 精度统计脚本
8.  benchmark工具源码地址：https://gitee.com/ascend/cann-benchmark/tree/master/infer

推理端到端步骤：

（1） git clone 开源仓https://github.com/ultralytics/yolov3/， 并下载对应的权重文件，修改**models/export.py**脚本生成onnx文件

```
git clone https://github.com/ultralytics/yolov3/
python3.7 models/export.py --weights ./yolov3.pt --img 416 --batch 1
```

（2）配置环境变量转换om模型

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --model=yolov3.onnx --framework=5 --output=yolov3_bs1 --input_format=NCHW --log=info --soc_version=Ascend310 --input_shape="images:1,3,416,416" --out_nodes="Reshape_219:0;Reshape_203:0;Reshape_187:0"
```

（3）解析数据集

下载coco2014数据集val2014和label文件**instances_valminusminival2014.json**，运行**parse_json.py**解析数据集

```
python3.7 parse_json.py
```

生成coco2014.names和coco_2014.info以及gronud-truth文件夹

（5）数据预处理

运行脚本preprocess_yolov3_pytorch.py处理数据集

```
python3.7 preprocess_yolov3_pytorch.py coco_2014.info yolov3_bin
```

（6）benchmark推理

运行get_coco_info.py生成info文件

```
python3.7 get_coco_info.py yolo_coco_bin_tf coco_2014.info yolov3.info
```

执行benchmark命令，结果保存在同级目录 result/dumpOutput_device0/

```
python3.7 get_coco_info.py yolo_coco_bin_tf coco_2014.info yolov3.info
```

（7）后处理

运行 bin_to_predict_yolo_pytorch.py 解析模型输出

```
python3.7 bin_to_predict_yolo_pytorch.py  --bin_data_path result/dumpOutput_device0/  --det_results_path  detection-results/ --origin_jpg_path /root/dataset/coco2014/val2014/ --coco_class_names /root/dataset/coco2014/coco2014.names --model_type yolov3  --net_input_size 416
```

运行map_cauculate.py统计mAP值

```
python3 map_calculate.py --label_path  ./ground-truth  --npu_txt_path ./detection-results -na -np
```

