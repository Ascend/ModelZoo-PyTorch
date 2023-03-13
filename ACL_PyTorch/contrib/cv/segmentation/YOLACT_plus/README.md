# YOLACT模型PyTorch离线推理说明

### 一、环境准备

#### 1、获取依赖库

```shell
pip3 install -r requirements.txt
git clone -b pytorch_1.5 https://github.com/ifzhang/DCNv2.git
cd DCNv2
python3.7 setup.py build develop
patch -p1 < ../dcnv2.diff
```

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
  ├── dcnv2.diff
  ├── DCNv2
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
  │   ├── prior.bin
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

官方训练完毕的权重文件：[yolact_plus_resnet50_54_800000.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Yolact_plus/PTH/yolact_plus_resnet50_54_800000.pth)

训练完毕的权重文件放于./weights目录下

#### 4、获取数据集

YOLACT模型使用Microsoft COCO 2017数据集进行训练及测试，下载数据集命令如下：

```bash
cd data/scripts
bash ./COCO.sh #获取测试数据集
```

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
bash test/pth2om.sh
```

注意：此处pth权重文件的路径应设为相对路径

#### 3、执行以下命令，开始离线推理

```bash
bash test/eval_acc_perf.sh
```

同时，benchmark工具会自动统计性能数据。

#### 4、在基准环境中执行以下命令，获取基准推理性能

onnx包含自定义算子，不能使用开源TensorRT测试性能数据，所以在基准服务器上在线推理测试性能数据。



### 三、评测结果

Yolact++不支持在bs16上离线推理，故在bs8上测试。

| 模型        | 在线推理精度              | 310离线推理精度                 | 基准性能     | 310性能   |
| ----------- | ----------------------- | ----------------------- | ---------- | --------- |
| YOLACT bs1 | mAP: box 34.94, mask 33.69 | mAP: box 34.90, mask 33.71 | 19.693fps  | 26.452fps |
| YOLACT bs8 | mAP: box 34.94, mask 33.69 | mAP: box 34.90, mask 33.71 | 16.377fps | 31.130fps |
