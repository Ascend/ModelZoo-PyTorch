# GaitSet

GaitSet是一个灵活、有效和快速的跨视角步态识别网络，迁移自https://github.com/AbnerHqC/GaitSet



## GaitSet Detail

1、将模型修改为使用apex的混合精度进行训练

2、将模型修改为使用DDP进行分布式训练

3、将模型修改为支持NPU环境下单p和多p的训练



## Requirements

1、`torch`、`torchvision`、`opencv-python`和`xarray`，`python3.7.5`

2、下载数据集CASIA-B，下载地址http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp，注意下载DatasetB



## Training

### 一、数据预处理

1、下载后的数据集内的压缩文件需要全部解压，解压后数据集内部的目录应为（`CASIA-B`数据集）：数据集路径/对象序号/行走状态/角度，例如`CASIA-B/001/nm-01/000/ `。



2、执行以下命令进行数据预处理，并转为二进制文件：（执行前需要按下面两步修改路径）

```bash
bash test/predeal.sh
```

预处理需要修改`predeal.sh`中的数据集路径。假设下载的数据集路径为`/home/CASIA-B/...`，初步处理后的数据集导出路径为`/root/CASIA-B-Pre`。则修改`predeal.sh`中含有`pretreatment.py`的一行如下：

```bash
python3.7.5 pretreatment.py --input_path='/home/CASIA-B' --output_path='/root/CASIA-B-Pre/'
```

修改后，把`config_1p.py`中的参数`dataset_path`也改为上面`output_path`所用的路径。例如导出路径为`/home/1/`则改为`"dataset_path": "/home/1"`。可以使用相对路径，起点为代码根目录。



> 如果不是使用python3.7.5，可修改`test`目录的脚本内的`python3.7.5`为自己的版本
>
> 执行任意`.sh`脚本提示`'\r': command not found`是文件编码问题，转为`linux`编码即可
>
> 预处理过程中提示大量`WARNING`属于正常现象。如果出现`ERROR`错误提示则可能路径设置有误、或要求中的库文件没有安装。由于`ERROR`提示等重新导出时，建议删除导出有误的文件后再导出。

运行时，首先初步处理后的数据集会在导出路径下生成。

随后，脚本会使用生成的数据集，在当前根目录下生成`CASIA-B-bin`文件夹，里面含有处理好的二进制格式的图片。之后，脚本会在当前根目录下生成以`.info`结尾的图片列表文件，用于推理。



### 二、推理

1、使用训练产生的ptm文件转为onnx和om文件：

(1) 如果已经训练过`Gaitset`，在`work/checkpoint/GaitSet`下生成了ptm文件：执行以下指令。当前目录下会生成`gaiset_submit.onnx`和`gaitset_submit.om`文件：

```bash
bash test/pth2om.sh
```

> 如果没有训练，可以使用源码自带的ptm进行推理，地址：https://github.com/AbnerHqC/GaitSet/tree/master/work/checkpoint/GaitSet。进入此地址下载里面的encoder.ptm后缀的文件



(2) 如果没有训练过，假设下载到本地的ptm路径为`/home/1.ptm`，可以先转onnx文件：

```bash
python3.7.5 pth2onnx.py --input_path='/home/1.ptm'
```

如果加载训练的ptm代数不为默认的80000代，例如为60000时需要设置`--iters`命令行参数：

```bash
python3.7.5 pth2onnx.py --input_path='/home/1.ptm' --iters=60000
```



转onnx生成`gaitset_submit.onnx`文件后，再执行下面的指令转om文件：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model=gaitset_submit.onnx --output=gaitset_submit --input_shape="image_seq:1,100,64,44" --log=debug --soc_version=Ascend310
```



2、在支持benchmark的环境下执行以下命令、并展示精度结果：

bs1：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=gaitset_submit.om -input_text_path=CASIA-B-bin.info -input_width=64 -input_height=64 -output_binary=True -useDvpp=False
```

bs16：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=gaitset_submit.om -input_text_path=CASIA-B-bin.info -input_width=64 -input_height=64 -output_binary=True -useDvpp=False
```



然后执行`eval_acc_perf.sh`：

```bash
bash test/eval_acc_perf.sh
```

或者在配置好了环境的前提下直接运行：

```bash
python3.7.5 -u test.py --iter=-1 --batch_size 1 --cache=True --post_process=True
```

参数`--iter`、`--cache`、`--post_process`为模型后处理固定参数不需修改。



原模型精度95.405%：

![](原精度.bmp)

转换后精度95.512%：

![](om精度.bmp)



