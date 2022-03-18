# 3DMPPE-ROOTNET模型PyTorch离线推理指导

### 环境准备 

安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt
```

### 安装开源模型代码  
```
git clone https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE.git  
cd 3DMPPE_ROOTNET_RELEASE
patch -p1 < ../3DMPPE_ROOTNET.patch
cd .. 
``` 
> branch: master

> commit id: a199d50be5b0a9ba348679ad4d010130535a631d

### 获取MuPoTS数据集  
下载 MuPoTS 解析数据 [[MuPoTS](https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE)]


### 获取推理工具
获取msame和benchmark工具 [[msame](https://gitee.com/ascend/tools/tree/master/msame)][[benchmark](https://gitee.com/ascend/cann-benchmark/tree/master/infer)]

将msame和benchmark.x86_64（或benchmark.aarch64）放到当前目录

### 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  
```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/root/datasets/
```
 **评测结果：**   
| 模型      | pth精度  | 310精度  | 性能基准    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| 3DMPPE-ROOTNET bs1  | AP_root: 31.87 | AP_root: 31.90 |  639.656fps | 664.718fps | 
| 3DMPPE-ROOTNET bs16 | AP_root: 31.87 | AP_root: 31.88 |  467.282fps | 817.480fps | 


