# Albert

base on [Albert-base-v2](https://github.com/lonePatient/albert_pytorch)

## 运行说明
- 下载[原始代码仓](https://github.com/lonePatient/albert_pytorch)
```
git clone https://github.com/lonePatient/albert_pytorch.git
cd albert_pytorch
git checkout 46de9ec
git apply ../albert.patch
cd ../
```
- 下载[数据集](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip) 并解压albert_pytorch/dataset/SST-2。

- 下载[预训练模型](https://drive.google.com/open?id=1byZQmWDgyhrLpj8oXtxBG6AA52c8IHE- ) 并解压到albert_pytorch/prev_trained_model/albert_base_v2。

- 下载[训练好的模型](https://pan.baidu.com/s/1G5QSVnr2c1eZkDBo1W-uRA )（提取码：mehp ）并解压到albert_pytorch/outputs/SST-2。

- 文件夹若无请新建

- `pip install -r requirements.txt`

- 获取 ais-bench 工具

  请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。。

## 模型转换

```bash
#设置环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
执行命令查看芯片名称（$\{chip\_name\}）

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

运行模型转换脚本转换OM

```shell
#pth转换为ONNX，此处以batchsize=32为例
python3.7 ./Albert_pth2onnx.py --batch_size=32 --pth_dir=./albert_pytorch/outputs/SST-2/ --onnx_dir=./outputs/

#使用onnxsim工具优化模型
python3.7 -m onnxsim ./outputs/albert_bs32.onnx ./outputs/albert_bs32s.onnx

#ONNX模型转换成OM文件
atc --input_format=ND --framework=5 --model=./outputs/albert_bs32s.onnx --output=./outputs/albert_bs32s --log=error --soc_version=Ascend${chip_name} --input_shape="input_ids:32,128;attention_mask:32,128;token_type_ids:32,128"

#ps：--soc_version处填入实际的芯片名称，例如Ascend310P3
```



## 数据预处理

```shell
# 前处理，pth_dir参数是预训练模型的路径
python3.7 Albert_preprocess.py --pth_dir=./albert_pytorch/outputs/SST-2/

# 生成bin文件，bin文件在bin_dir文件夹下，info和label文件在当前目录下
python3.7 gen_dataset_info.py --pth_dir=./albert_pytorch/outputs/SST-2/ --bin_dir=./bin/

# 拆分bin文件数据，并严格归并对应数据到bin1、bin2、bin3三个目录（跟模型的输入相匹配）
mkdir ./bin/bin1 ./bin/bin2 ./bin/bin3
mv ./bin/input_*.bin ./bin/bin1
mv ./bin/attention_*.bin ./bin/bin2
mv ./bin/token_*.bin ./bin/bin3
```



## 模型推理

```shell
#使用ais_bench对 om 模型进行推理
python3 -m ais_bench --model ./outputs/albert_bs32s.om --input ./bin/bin1,./bin/bin2,./bin/bin3 --output ./result/ --outfmt TXT --batchsize 32
#观察对应的性能打印结果throughput 1000*batchsize(32)/NPU_compute_time.mean(xx.xxx): xxx.xxx


# 竞品性能测试
trtexec --onnx=./outputs/albert_bs32s.onnx --fp16 --threads
```



## 模型后处理

```shell
#删除工具生成的中间文件
rm ./result/xxxx_xx_xx-xx_xx_xx（时间戳）/sumary.json

# 对数据进行后处理
python3.7 Albert_postprocess.py --dump_output=./result/xxxx_xx_xx-xx_xx_xx（时间戳）

#运行后会打印出对应的结果acc=xxx
#./result/xxxx_xx_xx-xx_xx_xx（时间戳）此目录根据实际值填写
```



## 结果

精度性能

| 模型      | pth精度  | 310精度  | 基准性能    | 310性能    | 310P3性能 |
| :------: | :------: | :------: | :------:  | :------:  | :------:  |
| Albert bs1  | acc:0.928 | acc:0.927  |  276.61fps | 231.39fps |  |
| Albert bs16 | acc:0.928  | acc:0.927 | 577.73fps | 300.83fps |  |
| Albert bs16 | acc:0.928 | acc:0.927 |  |  | 754.1fps |

