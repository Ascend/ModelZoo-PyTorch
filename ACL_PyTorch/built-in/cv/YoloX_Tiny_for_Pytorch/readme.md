###  YOLOX_tiny模型PyTorch离线推理指导

### 1. 环境准备

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 获取，修改与安装开源模型代码

```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git reset 3e2693151add9b5d6db944da020cba837266b --hard
pip3 install -v -e .
mmdetection_path=$(pwd)
cd ..
git clone https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
git reset 0cd44a6799ec168f885b4ef5b776fb135740487d --hard
pip3 install -v -e .
mmdeploy_path=$(pwd)
```

3. 点击链接https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth下载YOLOX-s对应的weights， 名称为yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth。放到mmdeploy_path目录下

4. 数据集

   使用coco2017数据集，请到https://cocodataset.org自行下载coco2017，放到/root/datasets目录(该目录与test/eval-acc-perf.sh脚本中的路径对应即可)，目录下包含val2017及annotations目录

5. [获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)

   将benchmark.x86_64或benchmark.aarch64放到当前YoloX_Tiny_for_Pytorch目录下

### 2. 离线推理

710上执行，执行时使npu-smi info查看设备状态，确保device空闲

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash test/pth2onnx.sh ${mmdetection_path} ${mmdeploy_path}
bash test/onnx2om.sh /root/datasets/val2017 ${mmdeploy_path}/work_dir/end2end.onnx yoloxint8.onnx Ascend710
bash test/eval_acc_perf.sh --datasets_path=/root/datasets --batch_size=64 --mmdetection_path=${mmdetection_path}
```

注意：量化要求使用onnxruntime版本为1.6.0

**评测结果：**

性能和精度保存在result.txt文件中

| 模型        | pth精度   | 710离线推理精度 | 性能基准  | 710性能 |
| ----------- | --------- | --------------- | --------- | ------- |
| YOLOX bs64 | box AP:0.337 | box AP:0.331 | fps 741 | fps 890 |



