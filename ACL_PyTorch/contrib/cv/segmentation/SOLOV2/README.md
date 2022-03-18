# SOLOV2模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip install -r requirements.txt  
```
说明：PyTorch选用开源1.9.0版本



2.获取，修改与安装开源模型代码  
安装mmcv
```
git clone https://github.com/open-mmlab/mmcv -b v0.2.16
cd mmcv
python setup.py build_ext
python setup.py develop
cd ..
```
获取SOLOv2代码
```
git clone https://github.com/WXinlong/SOLO.git -b master
cd SOLO
git reset --hard 95f3732d5fbb0d7c7044c7dd074f439d48a72ce5
patch -p1 < ../MMDET.diff
patch -p1 < ../SOLOV2.diff
pip install -r requirements/build.txt
pip install -v -e .
cd ..
```


3.获取权重文件  

请从[原始开源代码仓](https://github.com/WXinlong/SOLO)下载SOLOv2_R50_1x模型的权重文件

4.数据集     

数据集的获取请参考[原始开源代码仓](https://github.com/WXinlong/SOLO)的方式获取。请将val2017图片及其标注文件放入服务器/root/dataset/coco/文件夹，val2017目录存放coco数据集的验证集图片，annotations目录存放coco数据集的instances_val2017.json，文件目录结构如下：
```
root
├── dataset
│   ├── coco
│   │   ├── annotations
│   │   ├── val2017
```

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  
将benchmark.x86_64或benchmark.aarch64放到当前目录  

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  
```
#启动脚本内1-2行为pth2onnx，3-4行为onnx2om，脚本执行完成后会生成SOLOV2.onnx、SOLOV2_sim.onnx、solov2.om三个文件。
bash test/pth2om.sh  
#启动脚本内9-21行为前处理，用以获取处理后的图片信息与bin文件；23-33为获取图片info文件，为推理做准备；35-42行为benchmark推理；44-51行为后处理，同时会输出模型测评的精度；57-63行为打印om推理性能。
bash test/eval_acc_perf.sh --datasets_path=/root/datasets  
```
 **评测结果：**   
| 模型      | 在线推理精度  | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| SOLOV2 bs1  | mAP：34.0% | mAP:34.0% |  7.58fps | 9.877fps | 

备注：离线模型不支持多batch。
