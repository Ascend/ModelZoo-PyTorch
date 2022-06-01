#  CenterNet模型PyTorch离线推理指导

## 1 环境准备

1.安装必要的依赖，建议手动安装

```
pip3 install -r requirements.txt
```

2.获取，修改与安装开源模型代码

安装CenterNet，这里的CANN使用自带的环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh

git clone https://github.com/xingyizhou/CenterNet
cd CenterNet/src/lib/models/networks
rm -r DCNv2
rm -r pose_dla_dcn.py
git clone https://github.com/jinfagang/DCNv2_latest.git
mv DCNv2_latest DCNv2
cd DCNv2
rm -r dcn_v2.py
cd ../../../../../../
mv dcn_v2.py CenterNet/src/lib/models/networks/DCNv2
mv pose_dla_dcn.py CenterNet/src/lib/models/networks

cd CenterNet/src/lib/external
make
cd ../models/networks/DCNv2
python3 setup.py build develop
cd ../../../../../../
```

备注：将源码中DCNv2算子更新到DCNv2_latest，以支持pytorch1.8；按照上述步骤替换pose_dcn_dla.py文件与dcn_v2.py文件，以修改自定义算子，实现onnx的推理过程 (CANN版本为5.1.RC1)

另外，需要单独修改python环境中的utils.py文件，不同环境下具体路径有一定差异。手动将/usr/local/python3.7.5/lib/python3.7/site-packages/torch/onnx/utils.py下述部分做相应更改：

```python
if enable_onnx_checker and \
    operator_export_type is OperatorExportTypes.ONNX and \
        not val_use_external_data_format:
    # Only run checker if enabled and we are using ONNX export type and
    # large model format export in not enabled.
    # _check_onnx_proto(proto)
    pass
```
备注：在编译可变形卷积的时候可能出现编译不成功的情况，如果出现下面这类错误
```
error: ‘TORCH_CHECK_ARG’ was not declared in this scope
error: command '/usr/bin/g++' failed with exit code 1
```
需对DCNv2/src/cpu下的各个.cpp文件添加以下声明
```
#include <TH/TH.h>
```
除此之外，还需要对dcn_v2_cpu.cpp中141-142行处的TORCH_CHECK_ARG改为THArgCheck，
并将dcn_v2_psroi_pooling_cpu.cpp中的#include <ATen/ceil_div.h>注释掉

DCNv2/src/cuda下的各个.cu添加如下声明，.cu可用vim编辑修改
```
#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
```
同将dcn_v2_psroi_pooling_cuda.cu中的#include <ATen/ceil_div.h>注释掉，最后再重新执行python3 setup.py build develop进行编译，即可成功

3.获取权重文件

[ctdet_coco_dla_2x.pth](https://drive.google.com/open?id=1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT)，放在当前目录下

4.数据集
获取COCO数据集：[coco2017](https://cocodataset.org/#download)，下载其中val2017图片及其标注文件（[2017 Val images](http://images.cocodataset.org/zips/val2017.zip)，[2017 Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)），解压后放入/opt/npu/datasets/coco以及CenterNet/data/coco/路径下，其中val2017目录存放coco数据集的验证集图片，annotations目录存放coco数据集的instances_val2017.json，文件目录结构如下：

```
CenterNet
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── val2017
```
调用数据预处理脚本文件将数据转换为模型输入的数据
```
python3.7 CenterNet_preprocess.py /opt/npu/coco/val2017 ./prep_dataset
```

生成数据集info文件
```
python3.7 get_info.py bin ./prep_dataset ./prep_bin.info 512 512
```


4.模型转换.pth->.onnx
```
python3.7 CenterNet_pth2onnx.py ctdet_coco_dla_2x.pth CenterNet.onnx
```
执行ATC脚本完成onnx模型到om模型的转换

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model=CenterNet.onnx  --output=CenterNet_bs1 --input_format=NCHW --input_shape="actual_input:1,3,512,512" --out_nodes="Conv_1120:0;Conv_1123:0;Conv_1126:0" --log=info --soc_version=Ascend710
```
5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)
将benchmark.x86_64放到当前目录
```
chmod u+x benchmark.x86_64
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./CenterNet_bs1.om -input_text_path=./prep_bin.info -input_width=512 -input_height=512 -output_binary=True -useDvpp=False
```
## 2 离线推理测试

获取ctdet_coco_dla_2x.pth权重文件，放在主目录下，接着可以执行.sh完成整个推理流程

**1.pth转om：**

```
bash test/pth2om.sh
```
成功执行后会生成bs1和bs16对应的.om文件

**3.执行推理和评估脚本：**
```
bash test/eval_acc_perf.sh --datasets_path=/opt/npu/coco
```
**注： 若出现使用ATC、benchmark工具出现错误时，请参考推理指导书上的解决方案**

**评测结果：**

精度：

| 模型          | 官网pth精度 | t4在线推理精度| 310离线推理精度 | 310P离线推理精度  |
| ------------- | ----------- | -------------| -------- | -------- |
| CenterNet_bs1 | AP : 36.6   | AP : 36.6     | AP : 36.4 | AP : 36.4  |

性能：

| batchsize     | t4 | 310|  310P  |
| ------------- | ----------- | --------------- | -------- |
| 1 | 14.2857   | 19.11108       | 32.4865
| 4 | -   | 19.37716      | 35.7424
| 8 | -   | 19.50532       | 36.3168
| 16 | -   | 19.632       | 36.8814
| 32 | -   | 19.75924       | 37.1302

最优batch对比：

310P/t4 : 37.1302/14.2857>1.6 

310P/310: 37.1302/19.75924>1.2

备注：

1.原官网pth精度 AP : 37.4 是在线推理时keep_res(保持分辨率)的结果，但由于离线推理需要固定shape，故需要去掉keep_res(保持分辨率)。去掉keep_res(保持分辨率)后，跑在线推理精度评估得到  AP : 36.6 ，故以 AP : 36.6 作为精度基准

2.onnx因包含npu自定义算子dcnv2而不能推理，故使用在线推理测试性能

3.原模型在线推理中仅实现batchsize=1的精度测试和性能测试

