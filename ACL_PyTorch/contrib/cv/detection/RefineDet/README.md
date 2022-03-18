#RefineDet模型PyTorch离线推理指导

## 1 环境准备

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

```
pip3.7 install -r requirements.txt
```

2.获取代码和[权重文件](https://drive.google.com/file/d/1RCCTaNeby0g-TFE1Cvjm3dYweBiyyPoq/view?usp=sharing)，放到当前路径下

```
git clone https://github.com/luuuyi/RefineDet.PyTorch.git -b master
cd RefineDet.PyTorch
git reset --hard 0e4b24ce07245fcb8c48292326a731729cc5746a
patch -p1 < ../refinedet.patch

```

3.获取数据集,[VOC数据集](http://host.robots.ox.ac.uk/pascal/VOC)，可以通过下面的命令下载


```
sh data/scripts/VOC2007.sh
cd ../
```
4.获取[benchamrk](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)

把benchmark.x86_64 或者 benchmark.aarch64 放到当前目录


## 2 离线推理

pth转换为om
```
bash test/pth2om.sh 
```


精度,性能测试

```
 bash test/eval_acc_perf.sh --datasets_path=/root/datasets/VOCdevkit/
```




**评测结果：**
| 模型      | pth精度  | 310精度  |    基准性能    |310性能  |
| :------: | :------: | :------: | :------:  | :------:  | 
| RefineDet bs1  | [mAP:79.81%](https://github.com/luuuyi/RefineDet.PyTorch) | mAP:79.56%|  63.94fps | 101.24fps |
| RefineDet bs16 | [mAP:79.81%](https://github.com/luuuyi/RefineDet.PyTorch) |mAP:79.56% |  72.77fps | 136.8fps |




备注：

- nms放在后处理，在cpu上计算
- onnx转om时，不能使用fp16，否则精度不达标
  ```
  --precision_mode allow_fp32_to_fp16
  ```

