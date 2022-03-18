# PointRend模型PyTorch离线推理指导

## 1 环境准备


1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

```
pip3.7 install -r requirements.txt  
```

2.获取，修改与安装开源模型代码

```
git clone https://github.com/facebookresearch/detectron2
cd detectron2
git reset 861b50a8894a7b3cccd5e4a927a4130e8109f7b8 --hard
patch -p1 < ../PointRend.diff
cd ..
python3.7 -m pip install -e detectron2
```

3.获取权重文件

```
wget https://dl.fbaipublicfiles.com/detectron2/PointRend/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes/202576688/model_final_cf6ac1.pkl
```

4.数据集

[cityscape数据集](https://www.cityscapes-dataset.com/)

从官网获取gtFine_trainvaltest.zip和leftImg8bit_trainvaltest.zip，将这两个压缩包解压到创建的/root/datasets/cityscapes文件夹。  
[创建labelTrainIds.png](https://github.com/facebookresearch/detectron2/tree/main/datasets#expected-dataset-structure-for-cityscapes) ：
```
python3.7 createTrainIdInstanceImgs.py /root/datasets/cityscapes
```

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  
将benchmark.x86_64或benchmark.aarch64放到当前目录


## 2.离线推理

310上执行，执行时使npu-smi info查看设备状态，确保device空闲。

```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/root/datasets/cityscapes

```

评测结果

| 模型     | pth精度 | 310精度 | 性能基准 | 310性能                |
| -------- | ------- | ------- | -------- | ---------------------- |
| PointRend bs1 | [mIoU:78.86](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend#semantic-segmentation)   | mIoU:78.85   |  2.091fps |  0.845fps |

备注：  
1.onnx不支持grid_sample算子，参考mmcv的自定义算子grid_sample的测试等价代码bilinear_grid_sample进行替换  
2.由于分辨率大，内存的限制，模型暂不支持多batch