###  swin-s模型PyTorch离线推理指导

### 1. 环境准备

1. 安装依赖

```bash
pip install -r requirements.txt
```
修改torch的导出部分代码，将.../site-packages/torch/onnx/utils.py文件的_check_onnx_proto(proto)改为pass
2. 获取，修改与安装开源模型代码

```
git clone https://github.com/open-mmlab/mmcv
cd mmcv
MMCV_WITH_OPS=1 pip install -e .
cd ..
git clone https://github.com/open-mmlab/mmdetection
cd mmdetection
pip install -r requirements/build.txt
python setup.py develop

patch -p1 < ../swin.patch
cd ..
```

3. 将权重文件[mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth](https://github.com/open-mmlab/mmdetection/tree/master/configs/swin) 放到当前工作目录。

4. 数据集

   获取COCO数据集，并重命名为coco，放到/root/datasets目录

5. [获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)

   将benchmark.x86_64或benchmark.aarch64放到当前工作目录

### 2. 离线推理

310P上执行，执行时使npu-smi info查看设备状态，确保device空闲

```bash
bash test/pth2om.sh --batch_size=1
bash test/eval_acc_perf.sh --datasets_path=/root/datasets --batch_size=1
```

**评测结果：**

| 模型        | pth精度   | 310P离线推理精度 | 性能基准  | 310P性能 |
| ----------- | --------- | --------------- | --------- | ------- |
| YOLOX bs1 | box AP:48.2 | box AP:47.9 | fps 5.5 | fps 4.47 |
| YOLOX bs8 | box AP:48.2 | box AP:47.9 | fps - | fps 5.9 |



