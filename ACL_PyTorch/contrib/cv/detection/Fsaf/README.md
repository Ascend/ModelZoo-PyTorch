# Fsaf模型PyTorch离线推理指导

## 1 环境准备

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

```
pip3.7 install -r requirements.txt  
```  
   说明：PyTorch选用开源1.8.0版本    
2.获取，修改与安装开源模型代码

```
git clone https://github.com/open-mmlab/mmcv -b master 
git reset --hard 04daea425bcb0a104d8b4acbbc16bd31304cf168
cd mmcv
MMCV_WITH_OPS=1 pip3.7 install -e .
cd ..
git clone https://github.com/open-mmlab/mmdetection -b master
git reset --hard 604bfe9618533949c74002a4e54f972e57ad0a7a
cd mmdetection
patch -p1 < ../fsaf.diff
pip3.7 install -r requirements/build.txt
python3.7 setup.py develop
```
3.获取权重文件

[fsaf_r50_fpn_1x_coco-94ccc51f.pth](https://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_r50_fpn_1x_coco/fsaf_r50_fpn_1x_coco-94ccc51f.pth)

4.数据集  

[测试集](http://images.cocodataset.org/zips/val2017.zip)：coco/val2017/  
[标签](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)：coco/annotations/instances_val2017.json  

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  
  将benchmark.x86_64或benchmark.aarch64放到当前目录  
  
  
## 2 离线推理

310上执行，执行时使npu-smi info查看设备状态，确保device空闲

```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/root/datasets  
```

**评测结果：**

| 模型 | 官网pth精度                                                  | 310离线推理精度 | 基准性能 | 310性能 |
| ---- | ------------------------------------------------------------ | --------------- | -------- | ------- |
| Fsaf | [box AP:37.4%](https://github.com/open-mmlab/mmdetection/tree/master/configs/fsaf) | box AP:37.1%    | 8.9fps   | 40.0fps |
| Fsaf | [box AP:37.4%](https://github.com/open-mmlab/mmdetection/tree/master/configs/fsaf) | box AP:37.1%    | 6.9fps   | 40.0fps |

