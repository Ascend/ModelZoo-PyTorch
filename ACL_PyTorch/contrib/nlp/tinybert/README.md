TinyBERT模型PyTorch离线推理指导
======== 
本项目基于TinyBERT模型：[TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)

1.环境准备
===========
（1）安装必要的依赖
```bash
pip install -r requirements.txt
```
（2）获取，修改与安装开源模型代码
```
git clone https://github.com/huawei-noah/Pretrained-Language-Model.git  
cd ./Pretrained-Language-Model/TinyBERT
```
（3）获取权重文件

执行以下命令获取权重文件
```
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/nlp/TinyBert/%E3%80%90%E6%8E%A8%E7%90%86%E3%80%91%E6%A8%A1%E5%9E%8B%E6%96%87%E4%BB%B6.zip
```

（4）数据集

SST-2数据集可在开源仓获取。下载后只需提取SST-2数据集，然后重命名为glue_dir/SST-2，放到当前工作目录。运行如下命令即可：
```
wget https://ascend-pytorch-one-datasets.obs.cn-north-4.myhuaweicloud.com/train/zip/SST-2.zip
```

（5）获取推理工具

benchmark.x86_64/benchmark.aarch64请通过该[链接](https://gitee.com/ascend/cann-benchmark/tree/master/infer)
下载，msame请通过该 [链接](https://gitee.com/ascend/tools/tree/master/msame)
下载。


2.推理
===========
在310服务器上执行，执行时使用```watch npu-smi info```查看设备状态，确保device空闲。

- 若需要改变batch_size，请修改pth2onnx.sh脚本的eval_batch_size，并修改onnx2om.sh脚本中input_shape各输入的第0维数值；
- 若需要改变推理模式，请修改preprocess_data.sh和postprocess_data.sh的inference_tool.

（1）通过benchmark工具进行推理
--------------------
```
# 1.pth权重文件转onnx文件
bash ./test/pth2onnx.sh
# 输出：TinyBERT_sim.onnx文件

# 2.onnx文件转om模型（同时source环境变量）
bash ./test/onnx2om.sh
# 输出：TinyBERT.om文件

# 3.数据前处理
bash ./test/preprocess_data.sh
# 输出：input_ids, segment_ids, input_mask的二进制数据文件

# 4.获取info文件
bash ./test/get_info.sh
# 输出：TinyBERT.info文件

# 5.benchmark推理（同时获得fps性能数据。运行前使用chmod u+x ${inference tool}设置权限）
bash ./test/benchmark_inference.sh
# 输出：每个数据对应推理的logits（1×2数组，存放于bin文件），同时得到性能

# 6.数据后处理，同时获得测试集精度
bash ./test/postprocess_data.sh
# 屏幕打印精度

# 7. 将TinyBERT_sim.onnx上传至T4服务器，测试onnx性能
trtexec --onnx=TinyBERT_sim.onnx --shapes=input_ids:1x64,input_mask:1x64,segment_ids:1x64
# 得到GPU下的推理性能
```

（2）通过msame工具进行推理
--------------------
```
# 1.pth权重文件转onnx文件
bash ./test/pth2onnx.sh
# 输出：TinyBERT_sim.onnx文件

# 2.onnx文件转om模型（同时source环境变量）
bash ./test/onnx2om.sh
# 输出：TinyBERT.om文件

# 3.数据前处理
bash ./test/preprocess_data.sh
# 输出：input_ids, segment_ids, input_mask三个文件夹各放置872笔数据对应的二进制数据文件

# 4.msame推理
bash ./test/msame_inference.sh
# 输出：每个数据对应推理的logits值（1×2数组，存放于txt文件），同时得到性能

# 5.数据后处理，同时获得测试集精度
bash ./test/postprocess_data.sh
# 屏幕打印精度

# 6. 将TinyBERT_sim.onnx上传至T4服务器，测试onnx性能。bs指batch_size,ml指max_sequence_length
trtexec --onnx=TinyBERT_sim.onnx --shapes=input_ids:${bs}x${ml},input_mask:${bs}x${ml},segment_ids:${bs}x${ml}
# 得到GPU下的推理性能
```

结果
======
以下给出以benchmark作为推理工具的精度及性能数据：

|模型|官网pth精度|310推理精度|310性能|基准性能
|  ----  | ----  | ----|---- |---- | 
|<center>TinyBERT(bs1)|<center>无|<center>92.66|<center>1292.076|<center>972.157
|<center>TinyBERT(bs4)|<center>无|<center>92.66|<center>2857.572|<center>2850.363
|<center>TinyBERT(bs8)|<center>无|<center>92.66|<center>3431.772|<center>3325.615
|<center>TinyBERT(bs16)|<center>无|<center>92.66|<center>3772.492|<center>3415.359
|<center>TinyBERT(bs32)|<center>92.6|<center>92.66|<center>3873.08|<center>3746.713

备注：

- 该模型不支持动态shape，若需要更改推理的batch_size，请修改pth2onnx.sh/onnx2om.sh/benchmark_inference.sh或msame_inference.sh的batch_size参数；

- 性能单位:fps/card，精度为百分比；

- bs指batch_size.