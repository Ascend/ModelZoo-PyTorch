## Hubert模型离线推理指导

###  一、环境准备

Ascend环境: CANN 5.1.rc2

####  1. 创建conda环境

```bash
conda create -n hubert3.7.5 python=3.7.5
conda activate hubert3.7.5
pip install -r requirements.txt
```

####  2. 获取开源模型代码仓

```bash
git clone https://github.com/facebookresearch/fairseq.git -b main 
cd fairseq
git reset --hard 5528b6a38224404d80b900609463fd6864fd115a
patch -p1 < ../hubert.patch
cd ..
```

###  二、转 ONNX

1. 用户自行获得[test-clean](https://www.openslr.org/resources/12/test-clean.tar.gz)数据集，解压到./data/

```bash
bash ./hubert_data.sh
```

hubert_data.sh将对应数据集处理成tsv,ltr文件，同时获得pt文件
```
├── data
│     ├──LibriSpeech
|        ├──test-clean
│           ├──61
│           	├──70968
│                     ├──61-70968.trans.txt
│                     ├──61-70968-0000.flac
│                     ├──61-70968-0001.flac
│                     ├──......
│           	├──......	
│           ├──......	
|        ├──BOOKS.TXT
|        ├──CHAPTERS.TXT
|        ├──LICENSE.TXT
|        ├──README.TXT
|        ├──SPEAKERS.TXT
│     ├──pt
│        ├──hubert_large_ll60k_finetune_ls960.pt
│     ├──test-clean
│        ├──train.wrd
│        ├──train.tsv
│        ├──train.ltr 
```

2.执行转 ONNX 脚本

```bash
python3.7 pth2onnx.py --model_path ./data/pt/hubert_large_ll60k_finetune_ls960.pt --onnx_path ./hubert.onnx
```

###  三、转 OM

```bash
# 设置环境变量, 用户需使用自定义安装路径，指定为：
source /usr/local/Ascend/ascend_toolkit/set_env.sh
# 执行ATC参考命令
atc --framework=5 \
--model=hubert.onnx \
--output=hubert \
--input_format=ND \
--input_shape="source:1,580000" \
--soc_version=Ascend${chip_name} \
--log=error
```
说明：${chip_name}可通过 npu-smi info 指令查看，如下图标注部分。

![img](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

###  四、数据集预处理

####  1. 执行数据预处理脚本

```bash
mkdir -p ./pre_data/test-clean
python3.7 hubert_preprocess.py --model_path ./data/pt/hubert_large_ll60k_finetune_ls960.pt --datasets_tsv_path ./data/test-clean/train.tsv --datasets_ltr_path ./data/test-clean/train.ltr --pre_data_source_save_path ./pre_data/test-clean/source/ --pre_data_label_save_path ./pre_data/test-clean/label/
```

参数说明：

model_path：模型位置 ./data/pt/hubert_large_ll60k_finetune_ls960.pt

datasets_tsv_path：数据集tsv位置 ./data/test-clean/train.tsv

datasets_ltr_path：数据集ltr位置 ./data/test-clean/train.ltr

pre_data_source_save_path source：保存路径 ./pre_data/test-clean/source/

pre_data_label_save_path label：保存位置 ./pre_data/test-clean/label/
```
├── pre_data
│   ├──test-clean
|    ├──label 
|      ├──label0.bin 
|      ├──label1.bin
|      ├──......  
|    ├──source
|      ├──source0.bin 
|      ├──source1.bin 
|      ├──...... 
```


###  五、离线推理

1.准备 ais_bench 推理工具
查看[《ais_bench 推理工具使用文档》](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)，将工具编译后的压缩包放置在当前目录，解压工具包，安装工具压缩包中的whl文件：

```bash
pip3 install ./aclruntime-{version}-cp37-cp37m-linux_xxx.whl
pip3 install ./ais_bench-{version}-py3-none-any.whl
```

2.推理时，使用 npu-smi info 命令查看 device 是否在运行其它推理任务，提前确保 device 空闲

```bash
# 创建文件夹，存放推理结果文件
mkdir -p ./out_data/test-clean
python3.7 -m ais_bench --model ./hubert.om --input "./pre_data/test-clean/source/" --output "./out_data/test-clean/"
```
参数说明：

--model：om 模型路径

--input：预处理后的 bin 文件存放路径

--output：输出文件存放路径 

模型输出格式是bin，输出保存在"output"参数指定的文件夹中

3.执行数据后处理脚本

```bash
mkdir -p ./res_data/test-clean
python3.7 hubert_postprocess.py --model_path ./data/pt/hubert_large_ll60k_finetune_ls960.pt --source_json_path ./out_data/test-clean/*/sumary.json --label_bin_file_path ./pre_data/test-clean/label/ --res_file_path ./res_data/test-clean/
```
参数说明：

\--model_path ：表示模型路径

\-- source_json_path：表示离线推理输出所在的文件夹的json文件，路径为"./out_data/${datasets}/*/sumary.json 

--label_bin_file_path：表示正确答案的文件路径

--res_file_path：表示输出精度数据所在的文件名

4.性能测试 采用ais_bench纯推理模式：

```bash
python3.7 -m ais_bench --model ./hubert.om --loop=50 
```

推理性能吞吐率在“throughput”日志中显示。

**精度评测结果：**

推理数据的精度保存在./res_data/test-clean，性能信息保存在./out_data/test-clean/*/sumary.json

精度性能都是输入为1*580000的数据，即batchsize为1，帧数580000，当帧数不足580000时，后面补0

| 模型        | pth精度 | 310P离线推理精度 | 性能基准    | 310P性能   | 数据集     |
| ----------- | ------- | ---------------- | ----------- | ---------- | ---------- |
| hubert fp32 | 2.1341  | 2.1283           | 0.43599 fps | 4.47886fps | test-clean |


注：只支持batchsize为1
