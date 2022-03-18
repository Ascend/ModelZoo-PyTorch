# YOLACT模型PyTorch离线推理指导书

### 一、环境准备

#### 1、获取依赖库

```shell
pip3 install -r requirements.txt
```

其中，PyTorch建议使用1.8.0版本。

使用1.5.0版本PyTorch可以正常进行om导出，且om模型精度与性能正常；但导出的onnx模型文件无法使用trtexec在T4上进行测试。

使用1.8.0版本PyTorch则无以上问题。

#### 2、获取YOLACT源代码并更新

- 首先获取官方代码仓代码

  ```bash
  git clone https://github.com/dbolya/yolact.git
  ```

- 将无关文件删除，保留以下文件

  ```
  .
  ├── backbone.py
  ├── data
  │   ├── coco.py
  │   ├── config.py
  │   └── __init__.py
  ├── layers
  │   ├── box_utils.py
  │   ├── functions
  │   │   ├── detection.py
  │   │   └── __init__.py
  │   ├── __init__.py
  │   ├── interpolate.py
  │   └── output_utils.py
  ├── utils
  │   ├── augmentations.py
  │   ├── cython_nms.pyx
  │   ├── functions.py
  │   ├── __init__.py
  │   └── timer.py
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
  │   ├── coco.py
  │   ├── config.py
  │   └── __init__.py
  ├── env.sh
  ├── layers
  │   ├── box_utils.py
  │   ├── functions
  │   │   ├── detection.py
  │   │   └── __init__.py
  │   ├── __init__.py
  │   ├── interpolate.py
  │   └── output_utils.py
  ├── LICENSE
  ├── modelzoo_level.txt
  ├── README.md
  ├── requirements.txt
  ├── test
  │   ├── eval_acc_perf.sh
  │   ├── parse.py
  │   ├── perf_g.sh
  │   └── pth2om.sh
  ├── utils
  │   ├── augmentations.py
  │   ├── cython_nms.pyx
  │   ├── functions.py
  │   ├── __init__.py
  │   └── timer.py
  ├── weights
  │   └── pth2onnx.py
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

#### 1、执行离线推理前使用以下命令查看设备状态，确保device空闲

```bash
npu-smi info
```

#### 2、执行以下命令，生成om模型文件

```bash
cd test
./pth2om.sh pth权重文件的路径 生成onnx的文件名 生成om的文件名 输入batch_size
```

注意：此处pth权重文件的路径应设为相对路径

#### 3、执行以下命令，开始离线推理

```bash
./eval_acc_perf.sh 数据集图像路径 数据集标注路径 输入batch_size om文件路径 benchmark工具路径
```

同时，benchmark工具会自动统计性能数据

#### 4、在T4环境中执行以下命令，获取T4推理性能

```bash
./perf_g.sh 输入batch_size onnx文件路径
```

注意，如果使用1.5.0版本PyTorch，则导出的onnx可能无法使用trtexec工具进行性能测试。因此，这里建议使用1.8.0版本PyTroch。



### 三、评测结果

| 模型        | 官网精度                | 310精度                 | T4性能     | 310性能   |
| ----------- | ----------------------- | ----------------------- | ---------- | --------- |
| YOLACT_bs1  | box: 32.07，mask: 29.73 | box: 32.07, mask: 29.72 | 75.797FPS  | 84.014FPS |
| YOLACT_bs16 | box: 32.07，mask: 29.73 | box: 32.07, mask: 29.72 | 116.596FPS | 96.161FPS |

