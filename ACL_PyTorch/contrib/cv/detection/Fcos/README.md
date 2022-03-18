# FCOS模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```
说明：PyTorch选用开源1.8.0版本



2.获取，修改与安装开源模型代码  
安装mmcv  
{cuda_version}和{torch_version}根据自己的环境，参考https://github.com/open-mmlab/mmcv 的指导替换
```shell
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/{cuda_version}/{torch_version}/index.html
```
安装mmdetection
```
git clone https://github.com/open-mmlab/mmdetection -b master
cd mmdetection
git reset --hard dd0e8ede1f6aa2b65e8ce69826314b76751d4151
cp ../fcos_r50_caffe_fpn_4x4_1x_coco.py ./configs/fcos  
patch -p1 < ../fcos.diff
pip3.7 install -r requirements/build.txt
python3.7 setup.py develop
cd ..
```
修改pytorch代码去除导出onnx时进行检查，将/usr/local/python3.7.5/lib/python3.7/site-packages/torch/onnx/utils.py文件的_check_onnx_proto(proto)改为pass

3.获取权重文件  

[fcos基于mmdetection预训练的pth权重文件](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_r50_caffe_fpn_1x_4gpu_20190516-a7cac5ff.pth)  

4.数据集     
[coco2017](https://cocodataset.org/#download)，下载其中val2017图片及其标注文件，放入mmdetection/data/coco/文件夹，val2017目录存放coco数据集的验证集图片，annotations目录存放coco数据集的instances_val2017.json，文件目录结构如下：
```
mmdetection
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── val2017
```

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  
将benchmark.x86_64或benchmark.aarch64放到当前目录  

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  
```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/root/datasets  
```
 **评测结果：**   
| 模型      | 在线推理精度  | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| FCOS bs1  | mAP：35.0% | mAP:34.7% |  5.1fps | 29.498fps | 
| FCOS bs16 | mAP：35.0% | mAP:34.7% | 5.4fps | 38.299fps | 



