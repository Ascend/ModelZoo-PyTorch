# GFocalV2模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```
说明：PyTorch选用开源1.8.0版本



2.获取，修改与安装开源模型代码  
安装mmcv
```shell
git clone https://github.com/open-mmlab/mmcv -b v1.2.7
cd mmcv
MMCV_WITH_OPS=1 pip3.7 install -e .
cd ..
```
获取GFocalV2代码
```
git clone https://github.com/implus/GFocalV2.git -b master
cd GFocalV2
git reset --hard b7b355631daaf776e097a6e137501aa27ff7e757
patch -p1 < ../GFocalV2.diff
python3.7 setup.py develop
cd ..
```

3.获取权重文件  

[gfocalv2预训练的pth权重文件](https://drive.google.com/file/d/1wSE9-c7tcQwIDPC6Vm_yfOokdPfmYmy7/view?usp=sharing)  

4.数据集     
[coco2017](https://cocodataset.org/#download)，下载其中val2017图片及其标注文件，放入服务器/root/dataset/coco/文件夹，val2017目录存放coco数据集的验证集图片，annotations目录存放coco数据集的instances_val2017.json，文件目录结构如下：
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
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/root/datasets  
```
 **评测结果：**   
| 模型      | 在线推理精度  | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| GFocalV2 bs1  | mAP：41.0% | mAP:40.6% |  7.9fps | 12.071fps | 

备注：离线模型不支持多batch。