# PointNetCNN模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```

2.获取，修改与安装开源模型代码  
```
git clone https://github.com/hxdengBerkeley/PointCNN.Pytorch -b master   
cd PointCNN.Pytorch  
git reset 6ec6c291cf97923a84fb6ed8c82e98bf01e7e96d --hard 
patch -p1 < ../PointNetCNN.patch
cd ..
git clone https://gitee.com/Ronnie_zheng/MagicONNX.git
cd MagicONNX
pip install .
cd ..
cp  -r PointCNN.Pytorch/utils PointCNN.Pytorch/provider.py ./
```
3.获取权重文件  

将权重文件pointcnn_epoch240.pth放到当前工作目录  
```
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/PointNetCNN/pointcnn_epoch240.pth
```

4.数据集     
获取modelnet40_ply_hdf5_2048数据集，解压并放在./data目录下，
```
mkdir data
cd data
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/PointNetCNN/modelnet40_ply_hdf5_2048.zip
unzip -d  modelnet40_ply_hdf5_2048 modelnet40_ply_hdf5_2048.zip
cd ..
```

5.[获取msame工具](https://gitee.com/ascend/tools/tree/master/msame#https://gitee.com/ascend/tools.git)  
将msame工具放到当前工作目录  

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  
1-6行是 转onnx
7-14行是 onnx转om
```
bash test/pth2om.sh  
```
1-10行是 基本配置  
11-16行是 预处理  
17-22行是 使用msase工具推理  
23-30行是 使用benchmark工具推理  
38-43行是 精度统计  
44-50行是 性能统计  
```
bash test/eval_acc_perf.sh --datasets_path=/root/datasets  
```
 **评测结果：**   
|      模型       | pth精度 | 310离线推理精度 | 基准性能 |  310性能  |
| :-------------: | :-----: | :-------------: | :------: | :-------: |
| PointNetCNN bs1 | 82.61%  |     82.61%      |    31fps     | 27.3fps |

