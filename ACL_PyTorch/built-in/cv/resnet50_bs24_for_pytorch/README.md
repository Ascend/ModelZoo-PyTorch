文件作用说明：

1.  resnet_atc.sh：模型转换脚本,放于mmdeploy/work_dir目录下

2.  aipp.conf：配置文件,放于mmdeploy/work_dir目录下

3.  get_info.py： 数据集info生成脚本,放于mmdeploy/work_dir/atc目录下

4.  preprocess_resnet50_pytorch.py： 二进制数据集预处理脚本,放于mmdeploy/work_dir/atc目录下

5.  postprocess_resnet50_pytorch.py： 精度统计脚本,放于mmdeploy/work_dir/atc目录下

6.  requirements.txt: 版本要求

7. results.txt: 运行结果


推理端到端步骤：

（1） git clone 开源仓https://github.com/open-mmlab/mmclassification 和https://github.com/open-mmlab/mmdeploy ， 并下载对应的权重文件(https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar100_20210528-67b58a1b.pth ) 于mmdeploy目录下，使用mmdeploy的**depoly.py**脚本生成onnx文件

```shell
git clone https://github.com/open-mmlab/mmclassification.git
git clone https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
python3 tools/deploy.py ./configs/mmcls/classification_onnxruntime_dynamic.py /usr/local/mmclassification/configs/resnet/resnet50_b16x8_cifar100.py  resnet50_b16x8_cifar100_20210528-67b58a1b.pth /usr/local/mmclassification/demo/demo.JPEG --work-dir work_dir

注意: /usr/local的地址按照实际情况填写,生成的end2end.onnx文件位于work_dir目录下
```

（2）配置环境变量转换om模型

执行命令查看芯片名称（$\{chip\_name\}）。
   
```
npu-smi info
#该设备芯片名为Ascend310P3 （请根据实际芯片填入）
回显如下：
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 15.8         42                0    / 0              |
         | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
         +===================+=================+======================================================+
         | 1       310P3     | OK              | 15.4         43                0    / 0              |
         | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
         +===================+=================+======================================================+
```

```
cd work_dir

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 给soc_version传参数，该参数支持Ascend310和Ascend310P[1-4]，示例如下（按实际填写）
bash resnet_atc.sh Ascend310P3
        
运行成功后生成resnet50_bs24.om模型
```

（3）数据预处理

下载cifar数据集（http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz）, 放于atc目录下,运行脚本preprocess_resnet50_pytorch.py处理数据集，得到label文件

```
cd atc

python3 preprocess_resnet50_pytorch.py ./cifar-100-python/test ./bin_data

运行成功后,可以得到cifar100数据集的可视化是聚集pic,bin格式的数据集bin_data以及label文件img_label.txt
```

（4）生成数据集信息文件

运行**get_info.py**生成info文件

```
python3 get_info.py ./bin_data ./pre_data.info 32 32

运行成功后,可以得到pre_data.info数据集信息文件
```

（5）benchmark推理

执行benchmark命令，结果保存在同级目录 result/dumpOutput_device0/

```
./benchmark.aarch64 -model_type=vision -batch_size=24 -device_id=0 -input_text_path=./pre_data.info -input_width=32 -input_height=32 -om_path=./resnet_bs24.om -useDvpp=False --output_binary=False
```

（6）后处理

运行**postprocess_resnet50_pytorch.py**解析模型输出

```
python3 postprocess_resnet50_pytorch.py  /usr/local/result/dumpOutput_device0/  ./img_label.txt ./ result.json
```

（7）性能验证

```
./benchmark.aarch64 -batch_size=24 -om_path=resnet_bs24.om -round=1000 -device_id=0
```

（8）评测结果
运行map_cauculate.py统计mAP值

| 模型            | pth精度        | 310离线推理精度    | 性能基准    | 310P性能  |
|---------------|--------------|--------------|---------|---------|
| ResNet50 bs24 | Acc@1 79.90% | Acc@1 79.91% | 5411fps | 9346fps |

