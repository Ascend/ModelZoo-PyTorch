###  YOLOX_tiny模型PyTorch离线推理指导

### 1. 环境准备

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 第三方依赖

   需要从https://gitee.com/liurf_hw/om_gener 安装om_gener

3. 获取，修改与安装开源模型代码

```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git reset 3e2693151add9b5d6db99b944da020cba837266b --hard
pip3 install -v -e .
mmdetection_path=$(pwd)
cd ..
git clone https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
git reset 0cd44a6799ec168f885b4ef5b776fb135740487d --hard
pip3 install -v -e .
mmdeploy_path=$(pwd)
```

3. 点击链接https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox  下载YOLOX-s对应的weights， 名称为yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth。放到mmdeploy_path目录下
   
   ![yolox_tiny_download](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/built-in/cv/YoloX_Tiny_for_Pytorch/yolox_tiny.png)

4. 数据集

   使用coco2017数据集，请到coco官网自行下载coco2017，放到/root/datasets/coco目录(该目录与test/eval-acc-perf.sh脚本中的路径对应即可)，目录下包含val2017及annotations目录：

   ```
   |---- val2017
   |———— annotations                                      
   ```

5. [获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)

   将benchmark.x86_64或benchmark.aarch64放到当前YoloX_Tiny_for_Pytorch目录下

### 2. 离线推理

310P上执行，执行时使npu-smi info查看设备状态，确保device空闲

${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```bash
cp -r YoloX_Tiny_for_Pytorch ${mmdeploy_path}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash test/pth2onnx.sh ${mmdetection_path} ${mmdeploy_path}
bash test/onnx2om.sh /root/datasets/coco/val2017 ${mmdeploy_path}/work_dir/end2end.onnx yoloxint8.onnx Ascend${chip_name} # Ascend310P3
bash test/eval_acc_perf.sh --datasets_path=/root/datasets/coco --batch_size=64 --mmdetection_path=${mmdetection_path}
```

注意：量化要求使用onnxruntime版本为1.6.0；bbox_nms.py文件添加了BatchNMS自定义符号；deploy.py文件修改了headless参数为True，避免onnxruntime对onnx的推理；pytorch2onnx.py增加了enable_onnx_checker参数为false，避免对自定义算子的校验

**评测结果：**

性能和精度保存在results.txt文件中

| 模型        | pth精度   | 310P离线推理精度 | 性能基准  | 310P性能 |
| ----------- | --------- | --------------- | --------- | ------- |
| YOLOX bs64 | box AP:0.337 | box AP:0.331 | fps 741 | fps 890 |



