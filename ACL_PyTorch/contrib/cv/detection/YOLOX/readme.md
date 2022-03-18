###  YOLOX模型PyTorch离线推理指导

### 1. 环境准备

1. 安装依赖

```bash
pip3.7 install -r requirements.txt
```

2. 获取，修改与安装开源模型代码

```
git clone git@github.com:Megvii-BaseDetection/YOLOX.git -main
cd YOLOX
git reset 6880e3999eb5cf83037e1818ee63d589384587bd --hard
pip3.7 install -v -e .  # or  python3 setup.py develop
pip3.7 install cython
pip3.7 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
patch -p1 < ../YOLOX-X.patch
cd ..
```

3. 将权重文件[yolox_x.pth](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth)放到当前工作目录。

4. 数据集

   获取COCO数据集，并重命名为COCO，放到/root/datasets目录

5. [获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)

   将benchmark.x86_64或benchmark.aarch64放到当前工作目录

### 2. 离线推理

310上执行，执行时使npu-smi info查看设备状态，确保device空闲

```bash
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/root/datasets  
```

**评测结果：**

| 模型        | pth精度   | 310离线推理精度 | 性能基准  | 310性能 |
| ----------- | --------- | --------------- | --------- | ------- |
| yolox-x bs1 | map:51.2% | map:51.1%       | 60.739fps | 37.72144fps |



