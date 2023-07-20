## yolo系列后处理nms入图快速入门

### 第一步：
必须清楚onnx的nms算子输入bbox的可选择format
###### format1: upper_left_y, upper_left_x, lower_right_y, lower_right_x
###### format2: center_x, center_y, w, h
通过attributes："center_point_box"设置具体使用的format，"center_point_box": 0为format1，"center_point_box": 1为format2

### 第二步：
必须清楚原模型推理结果的bbox中的format，例如yolox_Megvii-BaseDetection仓库中的模型输出的bbox格式为center_x, center_y, w, h
是可以直接作为nms算子的输入的
    
### 第三步：
以Yolox为例，
方法一：将原后处理中nms之前所有的操作全部迁移至模型里，通过torch转出onnx，然后使用改图工具插入nms算子；
方法二：直接使用改图工具插入全部算子（要求对onnx算子非常熟悉）

修改模型结构逻辑如下：
```
原模型输出 -> nms之前的部分后处理  -> nms
                    ↓              ↓
               使用torch转出 -> 使用改图工具插入 -> 现模型输出
```
打patch
```
cp ./yolox_postprocess.py ./YOLOX/yolox/models/
cp ./modify.patch ./YOLOX
# 进入git clone下载的YOLOX目录下
cd YOLOX
git apply modify.patch
cd ..
export PYTHONPATH=$PYTHONPATH:./YOLOX
python3 YOLOX/tools/export_onnx.py -c ./yolox_s.pth -f YOLOX/exps/default/yolox_s.py --dynamic
```

### 第四步： 改图
在export导出的新onnx为基础，取其输出，分别gather出bboxes和scores，作为nms的输入项，保留模型原有输出，graph新增nms的输出作为整图输出，
其中add_nms_op.py中26、27行读取的"transpose333"算子节点，是开源模型输出节点前的transpose节点，如果模型有修改请以实际情况为准。
执行命令:
```
python3 add_nms_op.py
```
出现以下提示即可
```
yolox_modify.onnx add nms_op finished
```

### 第五步：
转出om
```
atc --model=yolox_modify.onnx --framework=5 --output=yolox_modify_bs4 --input_format=NCHW --input_shape='images:4,3,640,640' --soc_version=Ascend310P3 --device 0 --keep_dtype execeptionlist.cfg --precision_mode "allow_mix_precision"
```

### 第六步：
使用ais_bench进行推理
```
python -m ais_bench --m "./yolox_modify_bs4.om" --input "./prep_data/" --output ./result --device 0
```
其中各项参数含义与原README一致，路径也一致

### 第七步：
使用新的后处理脚本计算精度
```
python3 new_postprocess.py --dataroot=./coco --dump_dir="./result/2023_07_18-14_22_55"
```
      - 参数说明：

        - --dataroot：原始数据集所在路径。
        - --dump_dir：推理结果生成路径(以实际路径为准)。

最终精度计算结果为40.4，原仓精度为40.1
bs=4 模型性能值为377.36fps