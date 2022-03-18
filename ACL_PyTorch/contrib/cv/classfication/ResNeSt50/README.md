
# ResNeSt模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip3.7 install -r requirements.txt  
```


2.获取，修改与安装开源模型代码  

```
git clone https://github.com/zhanghang1989/ResNeSt   
```

3.获取权重文件  

[ResNeSt预训练pth权重文件](https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest50-528c19ca.pth)

```
wget https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest50-528c19ca.pth
```

4.数据集 

[ImageNet官网](http://www.image-net.org/) 

使用5万张验证集进行测试，图片与标签分别存放在/root/datasets/imagenet/val与/root/datasets/imagenet/val_label.txt。

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  

将benchmark.x86_64或benchmark.aarch64放到当前目录  

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  

```
bash test/pth2om.sh
bash test/eval_acc_perf.sh --datasets_path=/root/datasets 
```

 **评测结果：**   

| 模型      | 官网pth精度  | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| ResNeSt-50 bs1  | [top1:81.03%](https://github.com/zhanghang1989/ResNeSt) | top1:80.86% |  485fps | 643fps | 
| ResNeSt-50 bs16 | [top1:81.03%](https://github.com/zhanghang1989/ResNeSt) | top1:80.86% | 1052ps | 1088fps | 
