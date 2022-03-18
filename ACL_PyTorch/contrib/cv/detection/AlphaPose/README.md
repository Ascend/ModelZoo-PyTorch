# AlphaPose模型PyTorch离线推理指导

## 1 环境准备

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装
```
pip3.7 install -r requirements.txt
```

2.获取，修改与安装开源模型代码
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI/ 
make -j8
python3.7 setup.py install
cd -
git clone https://github.com/MVIG-SJTU/AlphaPose.git ./AlphaPose
cd AlphaPose
git reset ddaf4b99327132f7617a768a75f7cb94870ed57c --hard
git pull origin pull/592/head  # Ctrl-x退出
patch -p1 < ../AlphaPose.patch
python3.7 setup.py build develop
cd ..
```

3.获取权重文件

获取[fast_res50_256x192.pth](https://github.com/MVIG-SJTU/AlphaPose)，在工程目录下新建文件夹models，将pth文件放置到models文件夹内：

```
mkdir -p models
mv fast_res50_256x192.pth models
```
获取[yolov3-spp.weights](https://pan.baidu.com)放到对应文件夹下：

```
mkdir -p ./AlphaPose/detector/yolo/data
mv yolov3-spp.weights ./AlphaPose/detector/yolo/data
```

4.数据集
获取coco_val2017，新建data文件夹，数据文件目录格式如下：

```
mkdir data/
cp -r coco data/
```
文件目录如下:
```
data
└── coco
    ├── annotations
    └── val2017
```

5.[获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)  
将benchmark.x86_64或benchmark.aarch64放到当前目录  

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  
```
bash test/pth2om.sh
bash test/eval_acc_perf.sh --datasets_path=./data/coco
```
GPU机器上执行，执行时使用nvidia-smi查看设备状态，确保device空闲
```
bash perf_g.sh
```

 **评测结果：**
| 模型      | pth精度 | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| AlphaPose bs1 | mAP:71.73 | mAP:71.50 | 627.502fps | 330.596fps | 
| AlphaPose bs16 | mAP:71.73 | mAP:71.50 | 1238.543fps | 642.756fps | 
