TinyBERT模型PyTorch离线推理指导
======== 
本项目基于TinyBERT模型：[TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)

1.环境准备
===========
（1）安装必要的依赖
```bash
pip install -r requirements.txt
```

（2）数据集

本模型将使用到SST-2验证集的dev.tsv文件，通过[链接](https://cloud.easyscholar.cc/externalLinksController/chain/SST-2.zip?ckey=xpHPs51VaLh%2Fe6JlUc0mG6PEY%2BYHjBqk9LhT9WVYqL7eZu7WmXxb8m9Xxw6lf4ns)下载，获取成功后重命名为glue_dir/SST-2，放到当前工作目录即可。

（3）权重文件

从ModelZoo中获取权重文件包“SST-2_model.zip”，解压至当前目录

（4）安装ais_bench推理工具

请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。


2.推理
===========
在310P服务器上执行推理，执行时使用```watch npu-smi info```查看设备状态，确保device空闲。

--------------------
```
# 1.pth权重文件转onnx，并对onnx进行简化
bash ./test/pth2onnx.sh ${bs}
# 输出：若执行bash ./test/pth2onnx.sh 1，则生成TinyBERT_bs1.onnx和TinyBERT_sim_bs1.onnx
# 注：第一个参数代表批大小

# 2.onnx转om
bash ./test/onnx2om.sh ${bs} ${chip_name}
# 输出：若执行bash ./test/onnx2om.sh 1 710，则生成TinyBERT_bs1.om
# 注：第一个参数代表批大小；第二个参数代表处理器型号

# 3.数据前处理
bash ./test/preprocess_data.sh
# 输出：input_ids, segment_ids, input_mask三个文件夹各放置872笔数据对应的二进制数据文件

# 4.使用ais_bench推理工具推理
bash ./test/ais_inference.sh ${bs}
# 输出：若执行bash ./test/ais_inference.sh 1，则在当前路径的result文件夹内生成一个新的文件夹，同时在屏幕上打印出性能数据
# 注：第一个参数代表批大小

# 5.数据后处理，同时获得测试集精度
bash ./test/postprocess_data.sh ${filename}
# 输出：若步骤4中在/result路径下新生成的文件夹名为ais_infer_result_bs1，执行命令bash ./test/postprocess_data.sh ais_infer_result_bs1,则会在屏幕上打印出精度数据
# 注：第一个参数代表步骤4中新生成文件夹的名字

# 6. 将TinyBERT_sim.onnx上传至T4服务器，测试onnx性能
trtexec --onnx=TinyBERT_sim_bs${bs}.onnx --workspace=5000 --threads
# 输出：得到GPU下的推理性能
```

3.推理结果
======
以下给出以ais_bench作为推理工具的精度及性能数据：

|<center>模型|<center>官网pth精度|<center>310推理精度|<center>310P推理精度|<center>310性能|<center>310P性能|<center>T4性能|<center>310P/310|<center>310P/T4
|  ----  | ----  | ----|---- |---- | ---- | ---- | ---- | ---- | 
|<center>TinyBERT(bs1)|<center>无|<center>92.66|<center>92.32|<center>707.89|<center>1324.52|<center>972.16|<center>1.87|<center>1.36
|<center>TinyBERT(bs4)|<center>无|<center>92.66|<center>92.32|<center>2047.71|<center>3521.31|<center>2850.36|<center>1.72|<center>1.24
|<center>TinyBERT(bs8)|<center>无|<center>92.66|<center>92.32|<center>2883.62|<center>5871.86|<center>3325.62|<center>2.04|<center>1.77
|<center>TinyBERT(bs16)|<center>无|<center>92.66|<center>92.32|<center>3775.02|<center>8659.63|<center>3415.3590|<center>2.29|<center>2.54
|<center>TinyBERT(bs32)|<center>92.6|<center>92.66|<center>92.32|<center>4301.24|<center>10523.35|<center>3746.7130|<center>2.45|<center>2.81
|<center>TinyBERT(bs64)|<center>无|<center>92.66|<center>92.32|<center>4018.88|<center>11160.38|<center>4425.89|<center>2.78|<center>2.52
|<center>最优bs|<center>92.6|<center>92.66|<center>92.32|<center>4301.24|<center>11160.38|<center>4425.89|<center>2.59|<center>2.52

备注：

- 该模型不支持动态shape;

- 性能单位:fps/card，精度为百分比;

- bs指batch_size.