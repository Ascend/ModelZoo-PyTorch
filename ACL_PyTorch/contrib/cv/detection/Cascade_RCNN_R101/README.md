# Casacde_RCNN_R101模型PyTorch离线推理指导

## 1 环境准备 

1. 安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip3.7 install -r requirements.txt  
```

2. 获取，修改与安装开源模型代码  

```
git clone https://github.com/open-mmlab/mmdetection.git   
cd mmdetection  
git reset a21eb25535f31634cef332b09fc27d28956fb24b --hard
pip3.7 install -v -e .
patch -p1 < ../Cascade_RCNN_R101.patch   
cd ..
```

利用提供的change文件夹中的patch文件，完成补丁操作，命令参考如下示例,请用户根据安装包位置自行修改：
```
cd change
patch -p0 /usr/local/python3.7.5/lib/python3.7/site-packages/mmcv/ops/deform_conv.py deform_conv.patch
cd ../
```


3. 获取权重文件  

   从[LINK](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn)下载cascade_rcnn模型权重文件

4. 数据集    
   本模型使用coco2017的验证集验证 

5. [获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  
将benchmark.x86_64或benchmark.aarch64放到当前目录  
   

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  
```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=./
```
   

**评测结果：**   

|       模型        | 官网pth精度 | 310离线推理精度 | gpu性能 | 310性能  |
| :---------------: | :---------: | :-------------: | :-----: | :------: |
| Cascade_RCNN_R101 bs1 |  map:0.42  |    map:0.42    | 4.8task/s | 5.667fps |



