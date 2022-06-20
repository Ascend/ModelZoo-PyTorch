###  YOLOX模型PyTorch离线推理指导

### 1. 环境准备

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 获取，修改与安装开源模型代码

```
git clone -b master https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git reset 6b87ac22b8d9dea8cc28b9ce84909e6c311e6268 --hard

pip install -v -e .  # or  python3 setup.py develop
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.7.0/index.html
patch -p1 < ../YOLOX.patch
cd ..
```

3. 将权重文件[yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth)放到当前工作目录。

4. 数据集

   获取COCO数据集，并重命名为COCO，放到/root/datasets目录

5. [获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)

   将benchmark.x86_64或benchmark.aarch64放到当前工作目录

### 2. 离线推理

310P上执行，执行时使npu-smi info查看设备状态，确保device空闲

${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```bash
bash test/pth2om.sh --batch_size=1 --soc_version=Ascend${chip_name}
bash test/eval_acc_perf.sh --datasets_path=/root/datasets --batch_size=1
```

**评测结果：**

| 模型        | pth精度   | 310P离线推理精度 | 性能基准  | 310P性能 |
| ----------- | --------- | --------------- | --------- | ------- |
| YOLOX bs1 | box AP:50.9 | box AP:51.0 | fps 11.828 | fps 27.697 |
| YOLOX bs16 | box AP:50.9 | box AP:51.0 | fps 14.480 | fps 38.069 |



