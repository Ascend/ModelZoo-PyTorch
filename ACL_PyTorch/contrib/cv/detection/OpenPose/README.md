# OpenPose模型PyTorch离线推理指导
## 1 环境准备
### 1.1 安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt    
```
### 1.2 获取，安装开源模型代码
```
git clone https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch.git  
```
### 1.3 获取权重文件  
[OpenPose预训练pth权重文件](https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth)
```
wget https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth -P ./weights
```
### 1.4 数据集  
310服务器上可能已经下载好该数据集，若无，参考以下方法下载。  
[coco2017官网](https://cocodataset.org/#download)  
下载其中val2017图片及其标注文件，使用5000张验证集进行测试，图片与标注文件分别存放在/root/datasets/coco/val2017与/root/datasets/coco/annotations/person_keypoints_val2017.json。
文件目录结构如下，
```
root
├── datasets
│   ├── coco
│   │   ├── annotations
│   │   │   ├── captions_train2017.json
│   │   │   ├── captions_val2017.json
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   │   ├── person_keypoints_train2017.json
│   │   │   └── person_keypoints_val2017.json
│   │   ├── val2017
│   │   ├── annotations_trainval2017.zip
```
### 1.5 [获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  
将benchmark.x86_64或benchmark.aarch64放到当前目录，并更改权限
```
chmod 777 benchmark.x86_64
```
## 2 离线推理
### 2.1 测试
310上执行，执行时使npu-smi info查看设备状态，确保device空闲
```
bash test/pth2om.sh
bash test/eval_acc_perf.sh --datasets_path=/root/datasets
```
### 2.2 测评结果
|模型|pth精度(AP,%)|310精度(AP,%)|性能基准|310性能|
|----|----|----|----|----|
|OpenPose bs1|40|40.4|224.660fps|303.276fps|
|OpenPose bs16|40|40.4|339.973fps|444.908fps|

