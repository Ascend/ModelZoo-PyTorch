# SETR模型PyTorch离线推理指导

## 1 环境准备


1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

```
pip3.7 install -r requirements.txt  
```

2. 获取，修改与安装开源模型代码

```
# 手动编译安装1.2.7版本的mmcv
git clone git://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.2.7
MMCV_WITH_OPS=True python3.7 setup.py build_ext --inplace
MMCV_WITH_OPS=1 pip3.7 install -e .
cd ..

git clone https://github.com/fudan-zvg/SETR.git
cd SETR
git reset --hard 23f8fde88182c7965e91c28a0c59d9851af46858
patch -p1 < ../SETR.patch
pip3.7 install -e .
cd ..
```

3.获取权重文件

[SETR_Naive_768X768](https://drive.google.com/file/d/1f3b7I7IwKx-hcr16afjcxkAyXyeNQCAz/view?usp=sharing)

放到SETR/author_pth文件中

4.数据集

[cityscape数据集](https://www.cityscapes-dataset.com/)

从官网获取gtFine_trainvaltest.zip和leftImg8bit_trainvaltest.zip。将这两个文件解压后放到创建的/root/datasets/cityscapes文件夹中。
备注：执行cityscapes.py文件时会提示不存在train_extra文件，可以忽略。推理时不会用到，train、val文件夹处理好即可。

```
cd SETR
mkdir data
ln -s /root/datasets/cityscapes ./data
python3.7 tools/convert_datasets/cityscapes.py ./data/cityscapes --nproc 8 
cd ..
```

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)
将benchmark.x86_64或benchmark.aarch64放到当前目录


## 2.离线推理

310上执行，执行时使npu-smi info查看设备状态，确保device空闲

```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/root/datasets

```

评测结果

| 模型     | pth精度 | 310精度 | 性能基准 | 310性能                |
| -------- | ------- | ------- | -------- | ---------------------- |
| setr_naive_768x768 bs1 | [mIoU:77.36](https://github.com/fudan-zvg/SETR)   | mIoU:77.35   |  3.8497fps |  0.808fps |

备注:离线模型不支持多batch