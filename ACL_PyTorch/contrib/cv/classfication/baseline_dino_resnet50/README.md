# Dino_Resnet50模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```


2.获取开源模型代码  
```
git clone https://github.com/facebooksearch/dino   
```

3.获取权重文件  

 [获取权重文件](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/DINO/dino_resnet50_linearweights.pth) 
 [获取权重文件](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/DINO/dino_resnet50_pretrain.pth) 

4.数据集     
自行获取LSVRC2012验证集和标签文本

5.[获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)  
将benchmark.x86_64放到当前工作目录  

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  
```
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
bash test/pth2om.sh  
bash test/eval_acc_perf.sh
```
pth2om.sh文件第1到6行是转onnx，第8到20行是转om
eval_acc_perf.sh文件第24到54行是精度，第55到66行是性能
 **评测结果：**   
| 模型      | 官网pth精度  | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| dino_resnet50_bs1  | [top1: 75.3%](https://github.com/facebookresearch/dino#evaluation-linear-classification-on-imagenet) | top1: 75.27% | 891.845fps | 1521.508fps | 
| dino_resnet50_bs16 | [top1: 75.3%](https://github.com/facebookresearch/dino#evaluation-linear-classification-on-imagenet) | top1: 75.27% | 2003.345fps | 2406.68fps | 