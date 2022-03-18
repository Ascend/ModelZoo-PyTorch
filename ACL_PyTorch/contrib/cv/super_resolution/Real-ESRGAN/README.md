# Real-ESRGAN-baseline模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```

2.获取与安装开源模型代码  
```
git clone https://github.com/xinntao/Real-ESRGAN.git  
cd Real-ESRGAN 
git reset c9023b3d7a5b711b0505a3e39671e3faab9de1fe --hard
``` 

3.获取权重文件  

将权重文件[RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)放到experiments/pretrained_models/目录
```
 mkdir -p experiments/pretrained_models
 wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models 
```

4.数据集     
获取推理数据集：推理数据集代码仓已提供，并且放置在代码仓./Real-ESRGAN/inputs目录  

5.[获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)  
将benchmark.x86_64或benchmark.aarch64放到当前工作目录  

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  
```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=./Real-ESRGAN
```
 **评测结果：**   
| 模型      |    基准性能    | 310性能    |
| :------: | :------:  | :------:  | 
| Real-ESRGAN bs1  |  55.132fps | 139.502fps | 
| Real-ESRGAN bs16 | 72.923fps | 117.636fps | 

备注：  
加上TEST.NECK_FEAT "('before')" TEST.FEAT_NORM "('no')"导出的onnx可以进行离线推理  
不加TEST.NECK_FEAT "('before')" TEST.FEAT_NORM "('no')"导出的onnx转换的om精度与官网精度一致  



