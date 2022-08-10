# SiamFC模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip3.7 install -r requirements.txt  
```

2.获取，修改与安装开源模型代码  
```
代码地址：https://github.com/HonglinChu/SiamTrackers/tree/master/2-SiamFC/SiamFC-VID   
论文地址：https://arxiv.org/pdf/1606.09549.pdf
```
3.获取权重文件  

采用Ascend910上训练得到的权重文件[siamfc.pth](https://pan.baidu.com/s/1N3Igj4ZgntjRevsGA5xOTQ)，提取码：4i4l，放置于本代码仓./pth目录下

4.数据集     
[获取OTB2015数据集]([Visual Tracker Benchmark (hanyang.ac.kr)](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html))，并重命名为OTB，默认存放在/opt/npu目录下

5.[获取ais_infer工具]( https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)  
将ais_infer文件夹放到当前目录。

## 2 离线推理 

1.模型转化(pth-onnx-om)

pth-onnx
```
mkdir onnx
python3.7 pth2onnx.py siamfc.pth onnx/exemplar.onnx onnx/search.onnx

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/Ascend/driver/lib64/driver/
pip install sympy decorator
```

onnx-om, ${chip_name}可通过npu-smi info指令查看
![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)
```
atc --model=./onnx/exemplar.onnx --framework=5 --output=./om/exemplar_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,127,127" --log=debug --soc_version=Ascend${chip_name}
atc --model=./onnx/search.onnx --framework=5 --output=./om/search_bs1 --input_format=NCHW --input_shape="actual_input_1:1,9,255,255" --log=debug --soc_version=Ascend${chip_name}
```
2.离线推理
310P上执行，执行时使npu-smi info查看设备状态，确保device空闲  
```python3.7 wholeprocess.py datasets_path ./pre_dataset ./dataset_info 0```

> datasets_path参数根据数据集实际的存放位置而定，例如：OTB数据集存放位置为/opt/npu/OTB，则应设置参数--datasets_path=/opt/npu
> 第二个和第三个参数为数据处理中间保存变量存储位置， 最后一个参数为设备号

 **评测结果：**   

|    模型    |            t4 pth在线推理精度            |            310离线推理精度            |            310P离线推理精度            |
| :--------: | :------------------------------------------: | :------------------------------------------: | :------------------------------------------: |
| siamfc_bs1 | success_score: 0.576 precision_score: 0.767 | success_score: 0.571 precision_score: 0.760 | success_score: 0.572 precision_score: 0.762 |

3.性能测试
```
python3.7 get_perf_data.py ./pre_dataset1 ./pre_dataset2 ./dataset1.info ./dataset2.info
python3.7 ./ais_infer/ais_infer.py  --model ./om/exemplar_bs1.om --input pre_dataset1/ --device_id 0
python3.7 ./ais_infer/ais_infer.py  --model ./om/search_bs1.om --input pre_dataset2/ --device_id 0
```

| 模型 | T4性能 | 310性能 | 310P性能 | 310P(aoe)性能 | 310P/310 | 310P/T4 | 310P(aoe)/310 | 310P(aoe)/T4 |
| :------: | :------: | :------: | :------: | :------: | :------: | :------: |:------: | :------: |
| search_batch1| 95.16 | 862 | 932 | 1404 | 1.0812 | 9.7840| 1.6287 | 14.7541 |
| exemplar_batch1 | 1013 | 5677 | 2431 | 6135 | 0.4282 | 2.3998 | 1.0806| 6.0562|
| exemplar_batch4 | 378 | 11964| 13611| 15121| 1.1376| 36.0079| 1.2638| 40.0026|
| exemplar_batch8 | 194 | 13384| 17576|  | 1.3132| 90.5979| | |
| exemplar_batch16 | 97.8 | 13740| 14341|  | 1.0437| 146.6359| | |
| exemplar_batch32 | 56.6 | 13830| 13881|  | 1.0036| 245.2473| | |
| exemplar_batch64 | 26.1 | 14040| 14050|  | 1.0007| 538.3141| | |

> 由于该模型无法进行常规的离线测试，因而改为对测试集的每一帧进行测试，exemplar_bs1和search_bs1分别对应模型中的两个分支，它们所进行的操作不同。
>
> siamfc_bs1由exemplar_bs1和search_bs1这两部分组成，在评测精度时给出siamfc_bs1的精度，在评测性能时分别给出exemplar_bs1和search_bs1的性能。


