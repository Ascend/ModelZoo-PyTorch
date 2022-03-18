# Efficient-3DCNNs模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt
```

2.安装开源模型代码  
```
git clone https://github.com/okankop/Efficient-3DCNNs   
``` 
> branch: master

> commit id: d60c6c48cf2e81380d0a513e22e9d7f8467731af

3.获取权重文件  

[ucf101_mobilenetv2_1.0x_RGB_16_best.pth](https://drive.google.com/drive/folders/1u4DO7kjAQP6Zdh8CN65iT5ozp11mvE-H?usp=sharing)  

4.[获取UCF-101数据集](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)     
将UCF101.rar文件解压，重命名为ucf101，放在 /root/datasets/文件夹下

```
python3.7 Efficient-3DCNNs/utils/video_jpg_ucf101_hmdb51.py /root/datasets/ucf101/videos/ /root/datasets/ucf101/rawframes
python3.7 Efficient-3DCNNs/utils/n_frames_ucf101_hmdb51.py /root/datasets/ucf101/rawframes
```  
[获取json形式的annotation文件](https://github.com/okankop/Efficient-3DCNNs/tree/master/annotation_UCF101)   
将ucf101_01.json放到当前目录

5.获取[msame](https://gitee.com/ascend/tools/tree/master/msame)和[benchmark](https://gitee.com/ascend/cann-benchmark/tree/master/infer)工具    
将msame和benchmark.x86_64（或benchmark.aarch64）放到当前目录

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  
```
bash test/pth2om.sh  
bash test/eval_acc_perf.sh --datasets_path=/root/datasets/  
```
 **评测结果：**   
| 模型      | pth精度  | 310精度  | 性能基准    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| Efficient-3DCNNs bs1  | top1:81.100% top5:96.326% | top1:81.126% top5:96.299% |  619.767fps | 641.728fps | 
| Efficient-3DCNNs bs16 | top1:81.100% top5:96.326% | top1:81.126% top5:96.299% |  393.696fps | 744.432fps | 



