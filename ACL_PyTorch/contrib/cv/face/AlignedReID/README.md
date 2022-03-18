

# AlignedReID模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```

2.获取，安装开源模型代码至当前目录/AlignedReID下，打上修改部分的patch
```
git clone https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch.git 
cd AlignedReID-Re-Production-Pytorch
patch -p1 < ../all.patch
cd ..
```

3.获取权重文件  
[Market1501_AlignedReID_300_rank1_8441.pth](https://pan.baidu.com/s/1IcbfAZc2lrY7ioQ6uB4ySg)提取码：zvv5，放在当前目录/AlignedReID下

4.数据集     
[获取Market1501](https://drive.google.com/drive/folders/1CaWH7_csm9aDyTVgjs7_3dlZIWqoBlv4)下载链接中的文件夹market1501，放在当前目录/AlignedReID下，解压/market1501中的images.tar压缩文件
```
cd market1501
tar -xvf images.tar
```

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  
将benchmark.x86_64或benchmark.aarch64放在当前目录/AlignedReID下


## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  

导出不同bs的onnx模型并转化为om模型
```
cd ..
bash env.sh
bash test/pth2om.sh
```
对数据集进行前处理，生成info文件
```
bash test/data_preprocess.sh
```
进行推理，验证精度bs32
```
bash test/infer.sh
```


 **评测结果：**   
| 模型      | 910pth精度  | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| AlignedReID bs32  | [rank1:80.93%] | rank1:80.67% |  3098.88fps | 1869.44fps | 
| AlignedReID bs16  | - | - |  2809.43fps | 1659.87ps | 
| AlignedReID bs8  | - | - |  2552.07fps | 1428.40fps | 
| AlignedReID bs4  | - | - |  2200.87fps | 898.29fps | 
| AlignedReID bs1  | - | - |  1034.30fps | 315.84fps | 


备注：  
原仓库GPU复现精度为81%左右，910pth的精度是使用原仓库参数在NPU复现后的结果，其中310bs32的fps达到T4的60.3%，性能达标


