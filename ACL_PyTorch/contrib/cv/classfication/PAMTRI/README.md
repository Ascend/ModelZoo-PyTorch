# PAMTR模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  
```
pip3.7 install -r requirements.txt  
```


2.获取，修改与安装开源模型代码  
```
git clone https://github.com/NVlabs/PAMTRI -b master
cd MultiTaskNet
mv ./densenet.patch ./torchreid/models/ 
cd ./torchreid/models/
patch -p1 < densenet.patch  
```

3.获取权重文件  

将权重文件densenet121-a639ec97.pth放到当前工作目录（执行pth2onnx时会自动下载）
可以通过以下命令下载：
```
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/PAMTRI/densenet121-a639ec97.pth
```

4.数据集 
    
参考[github开源代码仓内数据集获取方法](https://github.com/NVlabs/PAMTRI)，将数据集放到./data目录。
下载[数据集的label文件](https://github.com/NVlabs/PAMTRI.git)，在PAMTRI/MultiTaskNet/data/veri/下获取label.csv文件。将数据集解压在./data/veri/目录下，应包含image_train、image_test、image_query三个数据集文件以及相应的.csv文件。

5.[获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)  
将benchmark.x86_64或benchmark.aarch64放到当前工作目录  

## 2 离线推理 

310上执行，执行时使npu-smi info查看设备状态，确保device空闲  
```
bash ./test/pth2om.sh  
#启动脚本内3-8行为pth2onnx，10-19行为onnx2om，脚本执行完成后会生成PAMTRI.onnx、PAMTRI_bs1.om、PAMTRI_bs16.om三个模型。
bash ./test/eval_acc_perf.sh --datasets_path=./data #datesets_path参数为指定数据集路径
#启动脚本内12-15行为前处理，将图片信息转换为二进制bin文件；17-24行为提取与汇总前处理生成的bin文件信息；32-58行为benckmark推理，会生成推理后的特征信息与推理性能文件；60-69行为后处理，对推理出的特征信息进行评测，生成精度信息文件；70-92行为打印推理性能和精度信息。
```
 **评测结果：**   
|           模型            |                         官网pth精度                          |    310离线推理精度    | 基准性能 |   310性能   |
| :-----------------------: | :----------------------------------------------------------: | :-------------------: | :------: | :---------: |
| PAMTRI bs1  | [mAP:68.64%](https://github.com/NVlabs/PAMTRI) | mAP:68.64% |  215.932fps  | 627.662fps |
| PAMTRI bs16 | [mAP:68.64%](https://github.com/NVlabs/PAMTRI) | mAP:68.64% | 810.976fps  | 833.668fps |
# 自检报告

```
# pth是否能正确转换为om
bash test/pth2om.sh
# 验收结果： OK

# 精度数据是否达标
# npu性能数据
bash test/eval_acc_perf.sh
# 验收结果： 是

# 在t4环境测试性能数据
bash test/perf_t4.sh
# 验收结果： OK

# 310性能是否符合标准： 是
bs1:310=2.9倍t4
bs16:310=1.03倍t4

```
