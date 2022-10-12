#  YOLACT模型PyTorch离线推理指导书

### 一、环境准备

#### 1、获取依赖库

```
pip3 install -r requirements.txt
```

其中，PyTorch建议使用1.8.0版本。

使用1.5.0版本PyTorch可以正常进行om导出，且om模型精度与性能正常；但导出的onnx模型文件无法使用trtexec在T4上进行测试。

使用1.8.0版本PyTorch则无以上问题。

#### 2、获取YOLACT源代码并更新

- 首先获取官方代码仓代码

  ```
  git clone https://github.com/dbolya/yolact.git
  ```

- 将无关文件删除，保留以下文件

  ```
  .
  ├── backbone.py
  ├── data
  │   ├── coco.py
  │   ├── config.py
  │   └── __init__.py
  ├── layers
  │   ├── box_utils.py
  │   ├── functions
  │   │   ├── detection.py
  │   │   └── __init__.py
  │   ├── __init__.py
  │   ├── interpolate.py
  │   └── output_utils.py
  ├── utils
  │   ├── augmentations.py
  │   ├── cython_nms.pyx
  │   ├── functions.py
  │   ├── __init__.py
  │   └── timer.py
  └── yolact.py
  ```

- 将本仓库代码拷贝至yolact目录下，并使用补丁YOLACT.patch复原

  ```
  patch -p1 < ./YOLACT.patch
  ```

  复原后，文件目录如下

  ```
  .
  ├── backbone.py
  ├── data
  │   ├── coco.py
  │   ├── config.py
  │   └── __init__.py
  ├── layers
  │   ├── box_utils.py
  │   ├── functions
  │   │   ├── detection.py
  │   │   └── __init__.py
  │   ├── __init__.py
  │   ├── interpolate.py
  │   └── output_utils.py
  ├── LICENSE
  ├── modelzoo_level.txt
  ├── README.md
  ├── requirements.txt
  ├── test
  │   ├── eval_acc_perf.sh
  │   ├── parse.py
  │   ├── perf_g.sh
  │   └── pth2om.sh
  ├── utils
  │   ├── augmentations.py
  │   ├── cython_nms.pyx
  │   ├── functions.py
  │   ├── __init__.py
  │   └── timer.py
  ├── weights
  │   └── pth2onnx.py
  ├── YOLACT.patch
  ├── YOLACT_postprocess.py
  ├── YOLACT_preprocess.py
  └── yolact.py
  ```

#### 3、获取权重文件

获取训练完毕的权重文件，建议放于./weights目录下

#### 4、获取数据集

YOLACT模型使用Microsoft COCO 2017数据集进行训练及测试。

在离线推理中仅使用测试数据集，测试图像为val 2017， 对应的标注文件为instances_val2017.json

#### 5、获取benchmark工具

获取benchmark.x86_64离线推理工具

### 二、离线推理

#### 1、执行离线推理前使用以下命令查看设备状态，确保device空闲并设置环境配置

```
npu-smi info
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

#### 2、执行以下命令，生成onnx模型文件

```
python3 ./weights/pth2onnx.py --trained_model=./weights/Yolact.pth --outputName=./weights/Yolact --dynamic=False --cann_version=5
```

注意：若需要设置为动态batch，则--dynamic=True；若CANN版本为6则--cann_version=6

#### 3、执行以下命令，生成om模型文件

```
atc --framework=5 --output=./weights/Yolact.onnx  --input_format=NCHW  --soc_version=Ascend310P3 --model=./weights/Yolact --input_shape="input.1:$batch_size,3,550,550"
```

注意：$batch_size：该参数根据自身需求设置，可设置为1，2，4，8，16等

#### 4、执行以下命令，进行数据集预处理

```
python3 ./YOLACT_preprocess.py --valid_images=$valid_images --valid_annotations=$valid_info
```

注意：$valid_images：传入验证集路径，如./coco/val_images/ ；$valid_info：传入instances_val2017.json所在路径,如./coco/annotations/instances_val2017.json

执行完成后，将在当前目录生成预处理后的bin文件及对应info文件

#### 5、执行以下命令，开始离线推理

```
./benchmark -model_type=vision -device_id=0 -batch_size=1 -om_path=./weights/Yolact.om
-input_text_path=./yolact_prep_bin.info -input_width=550 -input_height=550 -output_binary=True 
-useDvpp=False
```

注意：batch_size：该参数根据OM文件设置，可设置为1，2，4，8，16等；推理完成后，将在当前目录生产results文件夹

#### 6、执行以下命令，开始推理结果后处理，计算相关精度数据

```
python3 ./YOLACT_postprocess.py --valid_images=$1 --valid_annotations=$2
```

注意：$valid_images：传入验证集路径，如./coco/val_images/ ；$valid_info：传入instances_val2017.json所在路径,如./coco/annotations/instances_val2017.json

#### 4、在T4环境中执行以下命令，获取T4推理性能

```
./perf_g.sh 输入batch_size onnx文件路径
```

注意，如果使用1.5.0版本PyTorch，则导出的onnx可能无法使用trtexec工具进行性能测试。因此，这里建议使用1.8.0版本PyTroch。

### 三、评测结果

| 模型        | 官网精度                | 310精度                 | T4性能     | 310性能   |
| ----------- | ----------------------- | ----------------------- | ---------- | --------- |
| YOLACT_bs1  | box: 32.07，mask: 29.73 | box: 32.07, mask: 29.72 | 75.797FPS  | 84.014FPS |
| YOLACT_bs16 | box: 32.07，mask: 29.73 | box: 32.07, mask: 29.72 | 116.596FPS | 96.161FPS |