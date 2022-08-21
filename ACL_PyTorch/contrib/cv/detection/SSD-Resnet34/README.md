#  SSD-Resnet34模型PyTorch离线推理(cann==5.1.RC1)

## 一. 环境准备

### 1.通过requirements.txt 安装必要依赖

首先要先获取torch-1.5.0+ascend.post3.20210930-cp37-cp37m-linux_x86_64.whl，apex-0.1+ascend.20210930-cp37-cp37m-linux_x86_64.whl和tensor_fused_plugin-0.1+ascend-cp37-cp37m-linux_x86_64.whl这3个文件，获取方法如下：

获取torch-1.5.0+ascend.post3.20210930-cp37-cp37m-linux_x86_64.whl

x86架构：

```
wget https://ascend-pytorch-release.obs.cn-north-4.myhuaweicloud.com/run_pkg/20211018_FrameworkPTAdapter2.0.T308/torch-1.5.0%2Bascend.post3.20210930-cp37-cp37m-linux_x86_64.whl
```

ARM架构：

```
wget https://ascend-pytorch-release.obs.cn-north-4.myhuaweicloud.com/run_pkg/20211018_FrameworkPTAdapter2.0.T308/torch-1.5.0%2Bascend.post3.20210930-cp37-cp37m-linux_aarch64.whl
```

获取tensor_fused_plugin-0.1+ascend-cp37-cp37m-linux_x86_64.whl

x86架构：

```
wget https://ascend-pytorch-release.obs.cn-north-4.myhuaweicloud.com/run_pkg/20210423_TR5/whl_0423/tensor_fused_plugin-0.1%2Bascend-cp37-cp37m-linux_x86_64.whl
```

ARM架构：

```
wget https://ascend-pytorch-release.obs.cn-north-4.myhuaweicloud.com/run_pkg/20211018_FrameworkPTAdapter2.0.T308/torch-1.5.0%2Bascend.post3.20210930-cp37-cp37m-linux_aarch64.whl
```

获取apex-0.1+ascend.20210930-cp37-cp37m-linux_x86_64.whl

x86架构：

```
wget https://ascend-pytorch-release.obs.cn-north-4.myhuaweicloud.com/run_pkg/20211018_FrameworkPTAdapter2.0.T308/apex-0.1%2Bascend.20210930-cp37-cp37m-linux_x86_64.whl
```

ARM架构：

```
wget https://ascend-pytorch-release.obs.cn-north-4.myhuaweicloud.com/run_pkg/20211018_FrameworkPTAdapter2.0.T308/apex-0.1%2Bascend.20210930-cp37-cp37m-linux_aarch64.whl
```

在获得这3个.whl文件之后就使用命令直接运行：

x86架构：

```
pip install torch-1.5.0+ascend.post3.20210930-cp37-cp37m-linux_x86_64.whl
pip install apex-0.1+ascend.20210930-cp37-cp37m-linux_x86_64.whl
pip install tensor_fused_plugin-0.1+ascend-cp37-cp37m-linux_x86_64.whl
```

ARM架构：

```
pip install torch-1.5.0+ascend.post3.20210930-cp37-cp37m-linux_aarch64.whl
pip install apex-0.1+ascend.20210930-cp37-cp37m-linux_aarch64.whl
pip install tensor_fused_plugin-0.1+ascend-cp37-cp37m-linux_aarch64.whl
```

在运行上面的这条命令时，确保torch，apex和tensor_fused_plugin这3个.whl文件和requirements.txt在同一个目录下。

之后运行如下指令：

```
pip install -r requirements.txt
```

在运行完这条命令后，如果error中出现te0.4.0和schedule-search0.0.1相关信息，不需要去看，因为运行这个代码不需要用到，与本代码无关。

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
apt update
apt install libgl1-mesa-glx
```

之后再运行如上3条命令，本代码所需环境即安装完毕。

### 2. 从昇腾网站获取开源权重

<!-- 加载开源仓库：

```
git clone https://github.com/mlcommons/training_results_v0.7.git
```

进入开源代码仓，并打补丁，打补丁时确保补丁在开源代码仓路径的上一级：

```
cd training_results_v0.7/NVIDIA/benchmarks/ssd/implementations/pytorch/
patch -p1 <../ssd.patch
```

下载训练后的SSD权重文件：

```
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/detection/SSD-Resnet34/iter_183250.pt
```

下载基于搭建SSD模型的Resnet34模型的权重文件

```
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/detection/SSD-Resnet34/resnet34-333f7ec4.pth
``` -->
获取权重文件
- 浏览 [昇腾ModelZoo](https://www.hiascend.com/zh/software/modelzoo/models/detail/2/4845219a98e5426199b59b149d3a9a56) 网站，点击`立即下载`，并解压得到权重文件。

对于pth权重文件，统一放在新建models文件夹下，并将models文件夹放在当前文件夹下。

```
├── models
│    ├── iter_183250.pt
│    ├── resnet34-333f7ec4.pth
```

### 3. 获取测试数据集

本模型支持coco2017的val2017验证数据集，里面有5000张图片。用户可自行获取coco2017数据集中的annotations和val2017，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset），本模型将使用到coco2017数据集中的验证集及其标签文件instances_val2017.json, bbox_only_instances_val2017.json，标签文件bbox_only_instances_val2017.json是将coco2017中的原标签文件instances_val2017.json经过处理所得。

获得coco数据集的命令如下：

```
wget https://ascend-pytorch-one-datasets.obs.cn-north-4.myhuaweicloud.com/infer/zip/coco_2017_ssd_infer.zip
```

在本代码中我统一使用了coco这个名字来命名数据：

```
mv coco_2017_ssd_infer coco
```

获得新json标签文件的命令如下：

先给prepare-json.py增加权限，不然会出现权限不够的问题：

```
chmod -R 777 prepare-json.py
```

等增加完权限后再运行：

```
python3.7 prepare-json.py --keep-keys ${data_path}/coco/annotations/instances_val2017.json ${data_path}/coco/annotations/bbox_only_instances_val2017.json
```

第1部分${data_path}/coco/annotations/instances_val2017.json：这个是输入的json文件路径

第2部分${data_path}/coco/annotations/bbox_only_instances_val2017.json：这个是经过处理后输出的json文件路径。新的json文件命名一定要是bbox_only_instances_val2017.json，因为在代码中定义了运行json文件的名字。

${data_path}：代表数据集coco2017的路径

需要准备好的数据集部分：

```
├── coco
│    ├── val2017
│    ├── annotations
│         ├──instances_val2017.json
│         ├──bbox_only_instances_val2017.json
```

## 二. 离线推理

310上执行，执行时使npu-smi info查看设备状态，确保device空闲

执行代码脚本请在本工程代码文件夹下运行。

执行如下脚本生成om模型

1-25行是pth2onnx

29-43行是onnx2om

```
bash test/ssd_pth2om.sh
```

执行如下脚本进行数据预处理和后处理测试精度

```
bash test/ssd_eval_acc_perf.sh --data_path=/home/yzc
```

--data_path：coco2017数据集的路径

1-16行是加载数据集路径部分

19行是解决mlperf_logging包的调用问题

23-30行是处理json文件部分

32-40行是数据预处理部分

42-48行是生成info文件

50-67行是使用ais_infer进行离线推理的部分

70-84行是数据后处理评估精度部分

请用户在运行代码前，必须要先激活环境变量才能运行代码：

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

如果在运行代码的过程中，出现缺少.so库的问题，则需要再运行一遍上面输入的命令，再激活一次环境变量，即可解决问题。

另外，如果在运行过程中出现报出没有torchvision的错误，但实际已安装，请用户使用which python或者which python版本，查看python的路径是否在当前环境的路径下，请使用在当前环境路径下的相应python即可。

### 1.导出.onnx文件

−     使用iter_183250.pt导出onnx文件。

−     运行ssd_pth2onnx.sh可直接从pth转至om模型

运行ssd_pth2onnx.py脚本。

生成batchsize=1的onnx模型：

```
python3.7 ssd_pth2onnx.py --bs=1 --resnet34-model=./models/resnet34-333f7ec4.pth --pth-path=./models/iter_183250.pt --onnx-path=./ssd_bs1.onnx
```

生成batchsize=16的onnx模型：

```
python ssd_pth2onnx.py --bs=16 --resnet34-model=./models/resnet34-333f7ec4.pth --pth-path=./models/iter_183250.pt --onnx-path=./ssd_bs16.onnx
```

--bs：输入的batch_size大小

--resnet34-model：resnet34模型的pth权重文件路径

--pth-path：输入SSD模型的pth权重文件路径

--onnx-path：输出的onnx模型文件路径及onnx模型名字

执行上述步骤后，获得的输出有：

```
├── ssd_bs1.onnx
├── ssd_bs16.onnx
```

### 2.转om模型

设置环境变量

```
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

生成batchsize为1的om模型的命令如下。

```
atc --framework=5 --model=./ssd_bs1.onnx --output=./ssd_bs1 --input_format=NCHW --input_shape="image:1,3,300,300" --log=error --soc_version=ChipName
```

生成batchsize为16的om模型的命令如下。

```
atc --framework=5 --model=./ssd_bs16.onnx --output=./ssd_bs16 --input_format=NCHW --input_shape="image:16,3,300,300" --log=error --soc_version=ChipName
```

--framework：5代表ONNX模型。

--model：为ONNX模型文件输入的路径。

--output：输出的OM模型的路径。

--input_format：输入数据的格式。

--input_shape：输入数据的shape。

--log：日志级别。

--soc_version：处理器型号。

执行上述步骤后，获得的输出为：

```
├── ssd_bs1.om
├── ssd_bs16.om
```

### 3.数据预处理。

将原始数据集转换为模型输入的二进制数据。 

在进行数据预处理时，虽然coco2017的val2017验证集有5000张图片，但是实际上输出的只有4952张图片，因为在这过程中代码会剔除其中的48张图片。这一点请用户注意。

<!-- 在数据预处理之前先要声明mlperf_logging包的调用问题：

```
PYTHONPATH=../../../../../SIAT/benchmarks/resnet/implementations/tensorflow_open_src:$PYTHONPATH
``` -->

具体命令讲解：

```
python3.7 ssd_preprocess.py --data=${data_path}/coco --bin-output=./ssd_bin
```

--data：coco2017数据集的路径

--bin-output：经过预处理得到的bin文件路径

${data_path}：coco2017数据集的路径

执行上述步骤后，获得的输出为：

```
├── ssd_bin
│    ├── tensor([139]).bin
│    ├── ...
│    ├── tensor([581781]).bin
```

<!-- ### 4.生成数据集info文件

使用benchmark推理需要输入二进制数据集的info文件，用于获取数据集。使用get_info.py脚本，输入已经得到的二进制文件，输出生成二进制数据集的info文件。

具体命令讲解：

```
python3.7 get_info.py bin ./ssd_bin ssd.info 300 300
```

第一个参数为生成的数据集文件格式，

第二个参数为预处理后的数据文件相对路径，

第三个参数为生成的数据集文件名，

第四个和第五个参数分别为模型输入的宽度和高度。

执行上述步骤后，获得的输出为：

```
├── ssd.info
``` -->

### 5.使用Ais_Infer工具进行推理

Ais_Infer模型推理工具，其输入是om模型以及模型所需要的输入bin文件，其输出是模型根据相应输入产生的输出文件。

先后步骤顺序为：


− 由对batchsize=1的om模型进行ais_infer推理：

```
python ais_infer.py --model ${om_path}/ssd_bs1.om  --input /path/to/ssd_bin/ --output ${out_path}
```

− 由对batchsize=16的om模型进行ais_infer推理：

```
python ais_infer.py --model ${om_path}/ssd_bs16.om  --input /path/to/ssd_bin/ --output ${out_path}
```

- model：为om文件的路径

- input：为导出的数据集二进制文件ssd_bin/的路径

- output：为推理产生的结果的位置，注意这里的路径要填写一个已经存在的目录，会自动在这个目录中生成一个以日期命名的文件夹如`2022_08040-16_00_52`

${om_path}：om文件保存的目录
${out_path}：推理结果保存的目录

<!-- -input_height：输入高度

-useDvpp：为是否使用aipp进行数据集预处理，我这里不用

-output_binary：以预处理后的数据集为输入，benchmark工具推理om模型的输出数据保存为二进制还是txt。true为生成二进制bin文件，false为生成txt文件。 -->

<!-- -om_path：om模型文件路径。 -->

<!-- 执行./benchmark.x86_64工具请选择与运行环境架构相同的命令。参数详情请参见《 CANN 推理benchmark工具用户指南 》。 推理后的输出默认在当前目录result下。 -->

<!-- batchsize=1的om模型进行benchmark推理得到的bin文件输出结果默认保存在当前目录result/dumpOutput_device0；性能数据默认保存在result/ perf_vision_batchsize_1_device_0.txt。

batchsize=16的om模型进行benchmark推理得到的bin文件输出结果默认保存在当前目录result/dumpOutput_device1；性能数据默认保存在result/ perf_vision_batchsize_16_device_1.txt。 -->

该模型一个输入会对应两个输出，_0代表ploc的输出，_1代表plabel的输出。

执行以上命令后的输出：

```
├── 2022_08040-16_00_52
│    ├── tensor([139])_0.bin
│    ├── tensor([139])_1.bin
│    ├── ……
│    ├── tensor([139])_0.bin
│    ├── tensor([139])_1.bin
│    ├── ……

```

### 6.数据后处理

进行数据后处理时，也是需要调用同数据预处理一样的mlperf_logging包。因为在前面进行数据预处理时已经声明过了，所以可以不需要再进行声明了。

调用ssd_postprocess.py评测模型的精度：

batchsize=1的测试：

```
python ssd_postprocess.py --data=${data_path}/coco --bin-input=${output_path}
```

batchsize=16的测试：

```
python ssd_postprocess.py --data=${data_path}/coco --bin-input=${output_path}
```

--data：coco2017数据集的路径

--bin-input：数据预处理得到的bin文件。

${data_path}：coco2017数据集的路径
${output_path}：推理结果保存的路径，如`/xx/xx/2022_08040-16_00_52/`

### 7.评测结果：

| 模型              | 官网pth精度 | 310离线推理精度 | 性能基准   | 310性能    |
| ----------------- | ----------- | --------------- | ---------- | ---------- |
| SSD-Resnet34 bs1  | 23.000%     | 23.030%         | 482.627fps | 711.356fps |
| SSD-Resnet34 bs4 | 23.000%     | 23.030%         | 482.627fps | 825.516fps |
| SSD-Resnet34 bs8 | 23.000%     | 23.030%         | 482.627fps | 849.156fps |
| SSD-Resnet34 bs16 | 23.000%     | 23.030%         | 482.627fps |  862.468fps |
| SSD-Resnet34 bs32 | 23.000%     | 23.030%         | 482.627fps | 812.836fps |
| SSD-Resnet34 bs64 | 23.000%     | 23.030%         | 482.627fps | 810.368fps |
| Max(bs16) | 23.000%     | 23.030%         | - | 863.748fps |


| 模型              | 官网pth精度 | 310P离线推理精度 | 性能基准   | 310P性能    |
| ----------------- | ----------- | --------------- | ---------- | ---------- |
| SSD-Resnet34 bs1  | 23.000%     | 23.030%         | 482.627fps | 925.091fps |
| SSD-Resnet34 bs4 | 23.000%     | 23.030%         | 482.627fps | 1302.783fps |
| SSD-Resnet34 bs8 | 23.000%     | 23.030%         | 482.627fps | 1406.351fps |
| SSD-Resnet34 bs16 | 23.000%     | 23.030%         | 482.627fps | 1358.895fps |
| SSD-Resnet34 bs32 | 23.000%     | 23.030%         | 482.627fps |  1370.644fps |
| SSD-Resnet34 bs64 | 23.000%     | 23.030%         | 482.627fps | 914.474fps |
| Max(bs8) | 23.000%     | 23.030%         | - |  1406.351fps |
