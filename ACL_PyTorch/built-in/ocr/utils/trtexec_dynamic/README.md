## Tensorrt动态推理Api接口

### 安装

```
python3 setup.py install
```



### 模型转换

##### 调用示例

```angular2html
from tensorrt_dynamic.builder import Builder

builder = Builder(model_path, result_path, input_shapes, precision)
# builder.set_calibration_dataset(dataset)  
status = builder.build()
```
##### 参数说明

* model_path: onnx模型路径
* result_path: 模型转换结果保存路径
* precision: 模型精度，可选项："fp16"、"int8"
* input_shapes: 模型的动态输入，输入示例：

```
input_shapes = {
        "input1": {
            "min_shapes": [1, 3, 224, 224],
            "opt_shapes": [1, 3, 224, 224],
            "max_shapes": [32, 3, 448, 448]
        },
        "input2": {
            "min_shapes": [1, 3, 224, 224],
            "opt_shapes": [1, 3, 224, 224],
            "max_shapes": [32, 3, 448, 448]
        },
    }
# "input1"和"input2"为输入节点名，此处为模型具有多个输入场景
```

##### int8量化

precision设置为"int8"即可得到量化后的模型结果，但不保证精度；若要使用真实数据进行量化，可调用`builder.set_calibration_dataset(dataset) `，dataset格式为`List[List[np.ndarray]] `：

```
[ [input1, input2],
  [input1, input2],
       .....     
  [input1, input2] ]
```



### 模型推理

##### 调用示例

```
infer = Infer(trt_path)
for data in input_datas:
    result = infer.infer(data)
```

##### 参数说明

* trt_path: trt离线结果路径
* data: 输入数据，输入格式为`List[np.ndarray]`

##### 返回值

返回的结果为{输出节点名：输出值}`Dict[str, np.array]`