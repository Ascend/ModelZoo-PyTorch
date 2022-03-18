#  CenterNet模型PyTorch离线推理指导

## 1 环境准备

1.安装必要的依赖，建议手动安装

```
pip3 install -r requirements.txt
```

2.获取，修改与安装开源模型代码

安装CenterNet

```
source env.sh
git clone https://github.com/xingyizhou/CenterNet
cd CenterNet/src/lib/models/networks
rm -r DCNv2
rm -r pose_dla_dcn.py
git clone https://github.com/jinfagang/DCNv2_latest.git
mv DCNv2_latest DCNv2
cd DCNv2
rm -r dcn_v2.py
cd ../../../../../../
mv dcn_v2.py CenterNet/src/lib/models/networks/DCNv2
mv pose_dla_dcn.py CenterNet/src/lib/models/networks

cd CenterNet/src/lib/external
make
cd ../models/networks/DCNv2
python3 setup.py build develop
cd ../../../../../../
```

备注：将源码中DCNv2算子更新到DCNv2_latest，以支持pytorch1.5；按照上述步骤替换pose_dcn_dla.py文件与dcn_v2.py文件，以修改自定义算子，实现onnx的推理过程

另外，需要单独修改python环境中的utils.py文件，不同环境下具体路径有一定差异。手动将/usr/local/python3.7.5/lib/python3.7/site-packages/torch/onnx/utils.py下述部分做相应更改：

```python
            not val_use_external_data_format:
             # Only run checker if enabled and we are not using ATEN fallback and
             # large model format export in not enabled.
-            _check_onnx_proto(proto)
+            pass
```
3.获取权重文件

[ctdet_coco_dla_2x.pth](https://drive.google.com/open?id=1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT)，放在当前目录下

4.数据集
获取COCO数据集：[coco2017](https://cocodataset.org/#download)，下载其中val2017图片及其标注文件（[2017 Val images](http://images.cocodataset.org/zips/val2017.zip)，[2017 Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)），解压后放入/opt/npu/datasets/coco以及CenterNet/data/coco/路径下，其中val2017目录存放coco数据集的验证集图片，annotations目录存放coco数据集的instances_val2017.json，文件目录结构如下：

```
CenterNet
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── val2017
```

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)
将benchmark.x86_64放到当前目录

## 2 离线推理

CenterNet模型pth2onnx脚本由于算子暂不支持cpu，故只能在gpu运行，故将pth2om.sh拆为pth2onnx.sh和onnx2om.sh

**在gpu上：**

```
bash test/pth2onnx.sh
```

并将生成的CenterNet.onnx移到310上，路径为：{当前目录}/test

**在310上：**

**test目录下已经打包了一个正确的onnx，可解压后直接使用** 

```
unzip test/onnx.zip
```

```
bash test/onnx2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/opt/npu/datasets/coco
```

**评测结果：**

| 模型          | 官网pth精度 | 310离线推理精度 | 基准性能 | 310性能  |
| ------------- | ----------- | --------------- | -------- | -------- |
| CenterNet_bs1 | AP : 36.6   | AP : 36.4       | 23.25fps | 17.25fps |

备注：

1.原官网pth精度 AP : 37.4 是在线推理时keep_res(保持分辨率)的结果，但由于离线推理需要固定shape，故需要去掉keep_res(保持分辨率)。去掉keep_res(保持分辨率)后，跑在线推理精度评估得到  AP : 36.6 ，故以 AP : 36.6 作为精度基准

2.onnx因包含npu自定义算子dcnv2而不能推理，故使用在线推理测试性能

3.原模型在线推理中仅实现batchsize=1的精度测试和性能测试

