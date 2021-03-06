### Cascade Mask R-CNN UniFormer模型PyTorch离线推理指导

### 1. 环境准备

1. 安装依赖

CANN版本为5.1.RC1
```bash
pip install -r requirements.txt
```

2. 获取，修改与安装开源模型代码

```
git clone -b main https://github.com/Sense-X/UniFormer.git
cd UniFormer
git reset e8024703bffb89cb7c7d09e0d774a0d2a9f96c25 --hard

patch -p1 < ../uniformer.patch
cd object_detection
pip install mmcv-full==1.3.18 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.0/index.html
pip install -v -e .  # or python3 setup.py develop
cd ../..
```

3. 将权重文件[cascade_mask_rcnn_3x_ms_hybrid_base.pth](https://drive.google.com/file/d/13G9wc73CmS1Kb-kVelFSDlK-ezUBadzQ/view?usp=sharing)放到当前工作目录

4. 数据集

   获取COCO数据集，并重命名为coco，放到当前目录的data目录下

5. [获取msame工具](https://gitee.com/ascend/tools/tree/master/msame)

   将msame文件放到当前工作目录

### 2. 离线推理

310P上执行，执行时使npu-smi info查看设备状态，确保device空闲

${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```bash
bash test/pth2om.sh Ascend${chip_name}
bash test/eval_acc_perf.sh --datasets_path=data/coco
```

**评测结果：**

| 模型        | pth精度   | 310P离线推理精度 | 性能基准  | 310P性能 |
| ----------- | --------- | --------------- | --------- | ------- |
| UniFormer bs1 | bbox_mAP_50=72.5<br />segm_mAP_50=69.6 | bbox_mAP_50=72.0<br />segm_mAP_50=69.1 | 2.2 fps | 3.22 fps |
