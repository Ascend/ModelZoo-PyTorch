# GCNet模型PyTorch离线推理指导

## 1 环境准备 

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装  

```
pip3.7 install -r requirements.txt  
```



2.获取，修改与安装开源模型代码  

```
git clone https://github.com/open-mmlab/mmdetection
cd mmdetection
git reset --hard 6c1347d7c0fa220a7be99cb19d1a9e8b6cbf7544
pip3 install -r requirements/build.txt
python3 setup.py develop
cd ..
```



3.数据集     

使用COCO官网的coco2017的5千张验证集进行测试，请参考原始开源代码仓mmdetection中对公共数据集的设置

4.数据预处理

```python
python3 GCNet_preprocess.py --image_src_path=mmdetection/data/coco/val2017 --bin_file_path=val2017_bin --model_input_height=800 --model_input_width=1216
```

5.生成数据集info文件

5.1.生成jpg文件的输入info文件

```python
python3 gen_dataset_info.py jpg mmdetection/data/coco/val2017 coco2017_jpg.info
```

5.2.生成bin文件的输入info文件

```python
python3 gen_dataset_info.py bin val2017_bin coco2017.info 1216 800
```

6.[获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)  
将benchmark.x86_64或benchmark.aarch64放到当前工作目录  





## 2 模型转换

1.pth转onnx模型

a.使用GCNet.diff对mmdetection源码进行修改

```python
cp GCNet.diff ./mmdetection/
cd mmdetection
patch -p1 < ./GCNet.diff
cd ..
```

b.修改环境下onnx源码，除去对导出onnx模型检查

```python
vim /usr/local/python3.7.5/lib/python3.7/site-packages/torch/onnx/utils.py
```

修改文件的**_check_onnx_proto(proto)**改为**pass**，执行**:wq**保存并退出**。**

c.导出onnx文件

```
python3 mmdetection/tools/deployment/pytorch2onnx.py mmdetection/configs/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco.py mmdetection/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco_20200204-17235656.pth --output-file  GCNet.onnx --input-img mmdetection/demo/demo.jpg --test-img mmdetection/tests/data/color.jpg --shape 800 1216
```



2.onnx转om模型

a.配置环境变量。

```python
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/Ascend/driver/lib64
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/Ascend/driver/lib64/driver/
```

b.生成om模型

```
atc --framework=5 --model=GCNet.onnx --output=./GCNet_bs1 --input_shape="input:1,3,800,1216"  --log=error --soc_version=Ascend310P3
```



## 3 离线推理

1.使用benchmark工具推理

a.执行以下命令增加Benchmark工具可执行权限，并根据OS架构选择工具，如果是X86架构，工具选择benchmark.x86_64，如果是Arm，选择benchmark.aarch64 。

```python
chmod u+x benchmark.${arch}
```

b.执行推理

```python
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./GCNet_bs1.om -input_text_path=./coco2017.info  -input_width=1216 -input_height=800 -output_binary=True -useDvpp=False
```

2.数据后处理

```python
python GCNet_postprocess.py --bin_data_path=./result/dumpOutput_device0/ --test_annotation=coco2017_jpg.info --det_results_path=detection-results --annotations_path=/opt/npu/coco/annotations/instances_val2017.json --net_out_num=3 --net_input_height=800 --net_input_width=1216
```

3.精度验证

```python
python3 txt_to_json.py
```

4.对输出处理

```python
python3 coco_eval.py --ground_truth=mmdetection/data/coco/annotations/instances_val2017.json
```





6.GPU性能测试

onnx包含自定义算子，因此不能使用开源TensorRT测试性能数据，故在T4机器上使用pth在线推理测试性能数据

测评T4精度与性能

```
python tools/test.py configs/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco.py ./mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco_20200204-17235656.pth --eval bbox
python coco_eval.py
```





 **评测结果：**   

| 芯片型号 | Batch Size | 数据集  |  精度   |    性能     |
|:----:|:----------:|:----:|:-----:|:---------:|
| 310  |     1      | coco |       | 2.091fps  |
| 310p |     1      | coco | 0.610 | 12.031fps |
|  T4  |     1      | coco |       |  3.9fps   |

备注：  
1.GCNet的mmdetection实现不支持多batch。

2.onnx包含自定义算子，因此不能使用开源TensorRT测试性能数据，故在T4机器上使用pth在线推理测试性能数据。

