# Cascade-Mask-Rcnn模型PyTorch离线推理指导

## 1 环境准备

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

```
pip3.7 install -r requirements.txt  
```
   说明：PyTorch选用开源1.8.0版本    
2.获取，修改与安装开源模型代码

```sh
git clone https://github.com/facebookresearch/detectron2
python3.7 -m pip install -e detectron2
cd detectron2
patch -p1 < ../cascade_maskrcnn.patch 
cd ..
```
3.获取权重文件

[model_final_e9d89b.pkl](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)

4.数据集  

测试集：coco/val2017/  
标签：coco/annotations/instances_val2017.json  

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  
  将benchmark.x86_64或benchmark.aarch64放到当前目录  


## 2 离线推理

310上执行，执行时使npu-smi info查看设备状态，确保device空闲

```sh
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/opt/npu/datasets
```

**评测结果：**

| 模型              | 官网pth精度                                                  | 310离线推理精度 | 基准性能 | 310性能  |
| ----------------- | ------------------------------------------------------------ | --------------- | -------- | -------- |
| cascade_mask_rcnn | [mask AP:36.4%](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md) | Mask AP:36.27%  | 4.16 fps | 4.72 fps |