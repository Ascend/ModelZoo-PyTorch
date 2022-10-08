



# FaceDetection-DSFD模型-推理指导

FaceDetection-DSFD是通用场景下的人脸检测模型，采用FSSD+Resnet的网络结构，加入FEM 模块进一步增强不同感受野的目标特征，实现高准确率的人脸检测

## 1.环境准备

1.上传源码包到服务器任意目录并解压（如：/home/HwHiAiUser），根目录（/home/HwHiAiUser/DSFD）代码组织架构如下：

```
├── benchmark.aarch64            //离线推理工具（适用ARM架构） 
├── benchmark.x86_64             //离线推理工具（适用x86架构） 
├──bin_out                 //进行数据预处理生成的bin文件存放位置 
├──data       //模型配置文件 
├── eval_tools                        //进行模型准确率统计的脚本
        ├──setup.py    //依赖安装
        ├──dsfd_acc_eval   //模型准确性统计脚本
├──layers     //模型文件
├──models     //模型文件
├──onnx2om    //转化为om模型保存位置
├──result     //om模型推理结果
├──dsfd_preprocess.py         //数据预处理
├──dsfd_pth2onnx.py           //pth模型转化为onnx模型脚本
├──MagicONNX_optimization.py  //onnx模型算子调优脚本
├──README.md                  //说明文档
├──requirements.txt           //依赖项
```

2.在根目录下，安装必要的依赖：

```shell
pip install -r requirements.txt
```

3.获取推理图像数据集

[推理图像数据集](链接：https://pan.baidu.com/s/1KvpfjR0U8KUJnY7Gw5vLnQ  提取码：e3lu)

本模型使用WIDERFace数据集，请用户参照以上链接自行获取WIDER Face数据集，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset），数据目录请参考：

```
├──wider_face
    ├──annotations
    ├──ground_truth
    ├──image
    ├──labels
    ├──WIDER_val
```

4.通过模型补丁更新目录（不用下载开源仓，但是已参照开源仓进行修改）。

cd到/home/HwHiAiUser/DSFD/目录下，将源码包中的DSFD.patch文件移动到上级目录，保证补丁文件与DSFD/处于同一级目录中，然后执行以下命令进行打补丁：

```shell
patch -p0 < DSFD.patch
```

--p NUM 从文件名中去除 NUM 前导部分（同级目录的话NUM填0就可以），更多的选项可以通过命令 patch --help 查询，在执行该命令之后，会弹出一系列确认操作的选项，全部输入y并按回车即可。

5.进行数据预处理

数据预处理将原始数据集转换为模型输入的数据。需要先cd到DSFD/目录下，执行以下命令创建接收二进制数据文件的目录：

```shell
mkdir bin_out
```

使用“dsfd_preprocess.py”脚本进行数据预处理，脚本执行命令如下：

```shell
python3 dsfd_preprocess.py --src_path '${dataset_path}'
```

参数说明：--src_path：原始数据验证集（.jpeg）所在路径，运行后生成“info_result.info”；

## 2.模型转换

1.获取权重文件

[pth模型链接](链接：https://pan.baidu.com/s/1DKNAKusuSh8O_91xvpCtWw  提取码：i468) 下载后将dsfd.pth放在根目录下。

2.进行pth转onnx模型

cd到/home/HwHiAiUser/DSFD/路径下，使用“python3 dsfd_pth2onnx.py”脚本导出onnx文件。执行以下命令运行“dsfd_pth2onnx.py”脚本：

```shell
python3 dsfd_pth2onnx.py  --model_path ‘./dsfd.pth’ 
```

--model_path：权重文件路径。

执行后在/home/HwHiAiUser/DSFD路径下获得“dsfd.onnx”文件。

3.利用MagicONNX库对onnx模型进行算子优化

（1）安装MagicONNX库。

cd到/home/HwHiAiUser/DSFD路径下，依次执行以下命令：

```shell
git clone https://gitee.com/Ronnie_zheng/MagicONNX.git
cd MagicONNX
pip install .
```

全部运行成功后即在/home/HwHiAiUser/DSFD/MagicONNX路径下成功安装MagicONNX库。

（2）算子优化

在MagicONNX路径下，依次执行以下指令：

```shell
mv /home/HwHiAiUser/DSFD/dsfd.onnx /home/HwHiAiUser/DSFD/MagicONNX
mv /home/HwHiAiUser/DSFD/MagicONNX_test.py /home/HwHiAiUser/DSFD/MagicONNX、
```

将/home/HwHiAiUser/DSFD中的onnx模型与源码包中的MagicONNX_test.py移动到/home/HwHiAiUser/DSFD/MagicONNX路径下，并在该路径下执行：

```shell
python3 MagicONNX_test.py
```

即可得到算子优化后的ONNX模型文件test.onnx。

4.进行onnx模型转om模型

（1）配置环境变量：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

（2）执行以下ATC命令进行om模型转换：

bs1：

```shell
atc --framework=5 --model=./test.onnx --output=../onnx2om/dsfd_bs1 --input_format=NCHW --input_shape="image:1,3,224,224" --out_nodes="Reshape_884:0;Reshape_890:0;image:0;Reshape_896:0;Reshape_929:0;image:0" --log=debug --soc_version=${chip_name}
```

bs16：

```shell
atc --framework=5 --model=./test.onnx --output=../onnx2om/dsfd_bs16 --input_format=NCHW --input_shape="image:16,3,224,224" --out_nodes="Reshape_884:0;Reshape_890:0;image:0;Reshape_896:0;Reshape_929:0;image:0" --log=debug --soc_version=${chip_name}
```

参数说明：

- model：为ONNX模型文件。
- framework：5代表ONNX模型。
- output：输出的OM模型。
- input_format：输入数据的格式。
- input_shape：输入数据的shape。
- log：日志级别。
- soc_version：处理器型号。

其中soc_version可以通过npu-smi info指令进行查看。

运行成功后在/home/HwHiAiUser/DSFD/onnx2om/路径下生成om模型文件。

## 3.离线推理验证

1.增加Benchmark工具推理可执行权限。

执行以下命令增加Benchmark工具可执行权限，并根据OS架构选择工具，如果是X86架构，工具选择benchmark.x86_64，如果是Arm，选择benchmark.aarch64。

```shell
chmod u+x benchmark.x86_64
```

2.cd到/home/HwHiAiUser/DSFD目录下，执行以下命令：

```shell
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./onnx2om/dsfd_bs1.om -input_text_path=info_result.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False 
```

参数说明：

- model_type：模型类型。
- om_path：om文件路径。
- device_id：NPU设备编号。
- batch_size：参数规模。
- input_text_path：图片二进制信息。
- input_width：输入图片宽度。
- input_height：输入图片高度。
- useDvpp：是否使用Dvpp。
- output_binary：输出二进制形式。

推理后的输出默认在当前目录result下。

3.进行模型精度统计

（1）进入 eval_tools目录下，执行以下命令进行环境部署：

```shell
cd eval_tools
python setup.py build_ext --inplace
```

（2）cd到/home/HwHiAiUser/DSFD/ eval_tools/下，执行以下命令来创建接收“result.json”文件的文件夹：

```shell
mkdir infer_results
```

（3）调用“eval_tools/dsfd_acc_evalt.py”脚本与数据集标签“ground_truth”对比，可以获得Accuracy数据，结果保存在“result.json”中。

```shell
python3 dsfd_acc_eval.py -p '../result/dumpOutput_device0/' -g './ground_truth/'
```

第一个参数为生成推理结果所在路径，第二个参数为标签数据。

## 4.模型推理性能及精度

| Model | Batch Size | 310(FPS/Card) | 310P(FPS/Card) | 310P(AOE)(FPS/Card) | T4            (FPS/Card) | 310P(AOE)/310 | 310P(AOE)/T4 |
| ----- | ---------- | ------------- | -------------- | ------------------- | :----------------------- | ------------- | ------------ |
| DSFD  | 1          | 209.582       | 239.448        | 259.599             | 157.203538               | 1.238651      | 1.651356     |
| DSFD  | 4          | 280.548       | 465.076        | 681.274             | 279.193132               | 2.428369      | 2.440153     |
| DSFD  | 8          | 293.4656      | 503.229        | 781.203             | 336.479416               | 2.661992      | 2.321696     |
| DSFD  | 16         | 309.3548      | 495.672        | 827.772             | 386.976312               | 2.675801      | 2.139077     |
| DSFD  | 32         | 308.4068      | 479.551        | 840.612             | 422.835775               | 2.72566       | 1.988034     |



| Framework | Atlas  NPU Model | Server          | Container | Precision | Dataset    | Accuracy                                 | Ascend  AI Processor | NPU  Version         |
| --------- | ---------------- | --------------- | --------- | --------- | ---------- | ---------------------------------------- | -------------------- | -------------------- |
| PyTorch   | Atlas 300-3010   | Atlas  800-3010 | NA        | fp16      | WIDER FACE | Easy  Val AP: 0.9443  Medium Val AP: 0.9347  Hard  Val AP: 0.8645 | Ascend  310P         | Atlas  300-3010-32GB |