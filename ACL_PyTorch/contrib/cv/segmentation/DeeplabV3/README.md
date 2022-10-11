# DeeplabV3 模型PyTorch离线推理指导
## 推理
### 环境配置
```shell
pip install -r requirements.txt
pip install mmcv-full==1.3.15
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git checkout fa1554f1aaea9a2c58249b06e1ea48420091464d
pip install -e .
cd ..
```


### 转ONNX
[下载权重](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3)

* README.md文件中配置第一行最后一列model

#### 转onnx

```shell
python mmsegmentation/tools/pytorch2onnx.py \
mmsegmentation/configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py \
--checkpoint ./deeplabv3_r50-d8_512x1024_40k_cityscapes_20200605_022449-acadc2f8.pth \
--output-file deeplabv3.onnx --shape 1024 2048
```

#### 使用onnx-simplifier简化onnx

```python
python -m onnxsim deeplabv3.onnx deeplabv3_sim_bs1.onnx --input-shape="1,3,1024,2048" --dynamic-input-shape
```

### 转OM

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model=deeplabv3_sim_bs1.onnx --output=deeplabv3_bs1 --input_format=NCHW  \
--input_shape="input:1,3,1024,2048" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"
```


### 数据预处理
#### 前处理处理脚本 ./deeplabv3_torch_preprocess.py

```shell
python ./deeplabv3_torch_preprocess.py /opt/npu/cityscapes/leftImg8bit/val ./prep_dataset
```
读取./data/citiscapes/gtFine/val下的500张用于验证的图片，处理后保存为bin格式


#### 获取数据集信息文件

```shell
python ./gen_dataset_info.py bin ./prep_dataset ./deeplabv3_prep_bin.info 1024 2048
```

### 离线推理

将benchmark.x86_64放到目录

```shell
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=deeplabv3_bs1.om \ 
-input_text_path=./deeplabv3_prep_bin.info \
-input_width=1024 \
-input_height=2048 \
-output_binary=True \
-useDvpp=False
```

### 数据后处理

```shell
python ./deeplabv3_torch_postprocess.py --output_path=./result/dumpOutput_device0 --gt_path=/opt/npu/cityscapes/gtFine/val
```


### 评测结果

| 模型        | 官网精度                | 310精度      | T4性能     | 310性能   |
| ---------- | --------------------- | -------------| --------- | --------- |
| deeplabv3_bs1 |  mIoU 79.09        | mIoU 79.06   | 5.7787FPS | 3.1675FPS |