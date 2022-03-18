环境准备：

1.数据集路径
本模型数据集放在/opt/npu/，具体文件路径为：opt/npu/datasets
本模型支持cityscapes leftImg8bit的500张验证集。用户需要下载[leftImg8bit_trainvaltest.zip](http://www.cityscapes-dataset.com/downloads)和[gtFine_trainvaltest.zip](http://www.cityscapes-dataset.com/downloads)数据集，解压，将两个数据集放在/opt/npu/datasets/cityscapes/目录下。推荐使用软连接，可以节省时间，数据集目录如下。

```
|opt--npu--datasets
|          |-- cityscapes
|          |   |-- gtFine
|          |   |   |-- test
|          |   |   |-- train
|          |   |   |-- val
|          |   |-- leftImg8bit
|          |       |-- test
|          |       |-- train
|          |       |-- val
```

2.进入工作目录
cd Fast_SCNN

3.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装
pip3.7.5 install -r requirements.txt

4.获取，修改与安装开源模型代码
使用gitclone获取模型训练的代码，切换到tuili分支。

```
git clone https://gitee.com/wang-chaojiemayj/modelzoo.git
cd modelzoo
git checkout tuili
```

进入FastSCNN目录

```
cd ./contrib/PyTorch/Research/cv/image_segmentation/Fast_SCNN/
```

使用gitclone下载模型代码

```
git clone https://github.com/LikeLy-Journey/SegmenTron
```

由于onnx不支持AdaptiveavgPool算子，需要使用module.patch修改Fast_SCNN/SegmenTron/module.py。
将FastSCNN目录下的module.patch放到FastSCNN/SegmenTron目录下.
执行

```
cd ./SegmenTron
git apply module.patch
cd ..
```

5.获取权重文件
获取权重文件方法，可从Ascend modelzoo FastSCNN_ACL_Pytorch 模型压缩包获取

 md5sum:efc7247270298f3f57e88375011b52ee

6.数据预处理
在modelzoo/contrib/ACL_PyTorch/Research /cv/segmentation/FastSCNN目录创建软连接

```
ln -s /opt/npu/datasets datasets
```

运行Fast_SCNN_preprocess.py

```
python3.7.5  Fast_SCNN_preprocess.py
```

数据预处理的结果会保存在/opt/npu/prep_datset
预处理之后的二进制文件目录如下：
/opt/npu/prep_dataset/datasets/leftImg8bit/
/opt/npu/prep_dataset/datasets/gtFine/
在modelzoo/contrib/ACL_PyTorch/Research /cv/segmentation/FastSCNN目录下创建软连接

```
ln -s /opt/npu/prep_dataset prep_dataset
```

运行gen_dataset_info.py获取二进制输入文件的info信息

```
python3.7.5 gen_dataset_info.py
```

顺利运行会在当前目录下生成fast_scnn_prep_bin.info文件

6.获取benchmark工具
将benchmark.x86_64和benchmark.aarch64放在当前目录

7.310上执行，执行时确保device空闲

ascend-toolkit版本：5.0.2

onnx转出om

```
source env.sh（注意，latest是一个软连接，请将服务器中的/usr/local/Ascend/ascend-toolkit/latest 指向5.0.2版本的CANN包）
bash test/pth2om.sh
成功运行会生成fast_scnn_bs1.onnx，fast_scnn_bs4.onnx，fast_scnn_bs8.onnx，fast_scnn_bs16.onnx，fast_scnn_bs32.onnx;
fast_scnn_bs1.om，fast_scnn_bs4.om，fast_scnn_bs8.om，fast_scnn_bs16.om，fast_scnn_bs32.om文件。
(注意fast_scnn_bs32.onnx如果因为内存原因无法生成，也就无法导出fast_scnn_bs32.om。）
```

进行离线推理并进行精度、性能统计

```
bash test/eval_acc_perf.sh 
```
会自动对fast_scnn_bs1.om、fast_scnn_bs16.om、fast_scnn_bs4.om进行精度、性能的统计。（fast_scnn_bs16.om可能会因为内存原因无法进行离线推理，运行报错后会自动跳过）

8.在t4环境上将fast_scnn_bs1.onnx，fast_scnn_bs4.onnx，fast_scnn_bs8.onnx，fast_scnn_bs16.onnx，fast_scnn_bs32.onnx文件文件与perf_t4.sh放在同一目录

然后执行bash perf_t4.sh，执行时确保gpu空闲

