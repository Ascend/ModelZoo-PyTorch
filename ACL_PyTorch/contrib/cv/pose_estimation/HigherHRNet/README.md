# HigherHRNet模型PyTorch离线推理指导

## 环境准备

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

```
pip install -r requirements.txt  
```

2.获取，修改与安装开源模型代码

```
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
# Install into global site-packages
make install
python3 setup.py install --user

git clone https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation.git
cd HigherHRNet-Human-Pose-Estimation
git reset aa23881492ff511185acf756a2e14725cc4ab4d7 --hard
patch -p1 < ../HigherHRNet.patch
cd ..
```

3.获取权重文件

```
mkdir -p models
mv pose_higher_hrnet_w32_512.pth models
```

4.数据集 获取coco_val2017，新建data文件夹，数据文件目录格式如下：

```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

5.[获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)
将benchmark.x86_64或benchmark.aarch64放到当前目录

## 离线推理

310上执行，执行时使npu-smi info查看设备状态，确保device空闲

```
bash test/pth2om.sh
bash test/eval_acc_perf.sh --datasets_path=./data/coco
```

GPU机器上执行，执行时使用nvidia-smi查看设备状态，确保device空闲

```
bash test/perf_gpu.sh
```

**评测结果：**

| 模型            | pth精度  | 310离线推理精度 | 基准性能 | 310性能 |
| --------------- | -------- | --------------- | -------- | ------- |
| HigherHRNet bs1 | mAP:67.1 | mAP:67.1        | 109fps   | 193fps  |

