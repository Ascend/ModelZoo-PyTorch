# PointNet Onnx模型端到端推理指导

## 1. 模型概述
- 论文地址
- 代码地址

### 1.1 论文地址

[PointNet论文](https://arxiv.org/abs/1612.00593)

### 1.2 代码地址

[PointNet代码](https://github.com/fxia22/pointnet.pytorch)

## 2. 环境说明
1.安装必要的依赖

```
pip3.7 install -r requirements.txt
```
2.获取，修改与安装开源模型代码
```
git clone https://github.com/fxia22/pointnet.pytorch
cd pointnet.pytorch
git checkout f0c2430b0b1529e3f76fb5d6cd6ca14be763d975
patch -p1 < ../modify.patch
cd ..
```
## 3. 模型转换
- pth转onnx模型
- onnx转om模型
- 脚本运行
### 3.1 pth转onnx模型
1. 设置环境变量
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2. pth权重文件下载：
链接：https://pan.baidu.com/s/168Vk3C60iZOWrgGIBNAkjw 
提取码：lmwa
下载好后请放到PointNet工程目录下
3. 编写pth2onnx脚本，生成onnx文件
4. 执行pth2onnx.py脚本，生成具有动态shape的onnx模型文件
```shell
python3.7 pointnet_pth2onnx.py --model checkpoint_79_epoch.pkl --output_file pointnet.onnx
```
5. 由于在后期推理时，trt工具无法支持动态shape的onnx模型文件，所以先使用onnx-simplifier将输入优化成固定的shape，batch_size为1时如下：
```shell
python3.7 -m onnxsim pointnet.onnx pointnet_bs1_sim.onnx --input-shape="1,3,2500" --dynamic-input-shape
```
batch_size为16时如下：
```shell
python3.7 -m onnxsim pointnet.onnx pointnet_bs16_sim.onnx --input-shape="16,3,2500" --dynamic-input-shape
```
batch_size为4,8,32时类似
6. 利用fix_conv1d.py对模型进行优化，提升性能
先准备必要的环境：
```shell
git clone https://gitee.com/zheng-wengang1/onnx_tools.git
cd onnx_tools && git checkout cbb099e5f2cef3d76c7630bffe0ee8250b03d921
cd ..
```
对batch_size为1的onnx模型进行优化：
```shell
python3.7 fix_conv1d.py pointnet_bs1_sim.onnx pointnet_bs1_sim_fixed.onnx
```
batch_size为4,8,16,32类似
### 3.2 onnx转om模型
1. 使用atc将onnx模型转换为om模型文件， batch_size为1时：
```shell
atc --framework=5 --model=./pointnet_bs1_sim_fixed.onnx --output=./pointnet_bs1_fixed --input_shape="image:1, 3, 2500" --soc_version=Ascend310  --log=error > atc.log
```
若要生成batch_size为16的om模型，对应命令为：
```shell
atc --framework=5 --model=./pointnet_bs16_sim_fixed.onnx --output=./pointnet_bs16_fixed --input_shape="image:16, 3, 2500" --soc_version=Ascend310  --log=error > atc.log
```
batch_size为4,8,32时类似
### 3.3 脚本运行
运行如下脚本即可完成上述过程
```shell
bash test/pth2om.sh
```
## 4. 数据预处理
- 数据集获取
- 数据集预处理
- 生成数据信息文件
### 4.1 数据集获取
1. 进入到数据集下载脚本的文件夹下
```shell
cd test
```
2. 执行数据集下载脚本
```shell
source download.sh
```
3. 返回上级目录并新建一个文件夹，将下载好的数据移动到此文件夹下
```shell
cd ..
mkdir data
mv shapenetcore_partanno_segmentation_benchmark_v0 ./data/
```
### 4.2 数据集预处理
1. 仿照github官网训练预处理方法处理数据，编写预处理脚本pointnet_preprocess.py，由于这里数据集是非图片类文件，所以使用二进制的输入
2. 执行预处理脚本，生成数据集预处理后的bin文件。第一个参数为数据集存放的位置，第二个参数为预处理后所有二进制文件保存的路径，第三个参数为batch_size
```shell
python3.7 pointnet_preprocess.py data/shapenetcore_partanno_segmentation_benchmark_v0 ./bin_file batch_size=1
```
生成batch_size为16的数据集预处理后的bin文件：
```shell
python3.7 pointnet_preprocess.py data/shapenetcore_partanno_segmentation_benchmark_v0 ./bin_file_bs16 batch_size=16
```
## 5. 离线推理
- 特别说明：在这里由于PointNet模型用于分类时的输入shape原因不能使用benchmark工具在训练数据上做推理，只能使用纯推理方式。所以使用另外一个名为msame的工具在训练数据上做推理。而msame工具不能设置推理时的batch_size，所以只好对bs1和bs16各自生成对应的二进制文件，然后用于推理。而且由于样本量不是16的倍数，所以对于bs16的情况会遗留一些样本没办法推理，导致精度上和bs1有一点小差距。
### 5.1 msame工具
[msame使用说明](https://gitee.com/ascend/tools/tree/master/msame)
下载msame并解压到PointNet目录下，本模型使用的是Ascend-5.0.2，所以使用如下下载链接：
https://obs-book.obs.cn-east-2.myhuaweicloud.com/cjl/msame.zip

#### 5.1.1 设置环境变量
(如下为设置环境变量的示例，请将/home/HwHiAiUser/Ascend/ascend-toolkit/latest替换为Ascend 的ACLlib安装包的实际安装路径。)

export DDK_PATH=/home/HwHiAiUser/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=/home/HwHiAiUser/Ascend/ascend-toolkit/latest/acllib/lib64/stub
#### 5.1.2 运行编译脚本
cd msame/
chmod 777 build.sh
./build.sh g++ out
### 5.2 离线推理
1. 在PointNet工程下新建一个文件夹存放推理结果
```shell
mkdir res_data
```
2. 针对batch_size为1的情况，运行如下命令进行离线推理：
```shell
./msame/out/msame --model "pointnet_bs1_fixed.om" --input "bin_file" --output "res_data/bs1_out/" --outfmt TXT --device 1
```
3. 进入到res_data目录下，对上一步推理输出的结果文件夹重命名为bs1_out
4. batch_size为16时的推理，注意将输出的结果文件夹重命名为bs16_out
```shell
./msame/out/msame --model "pointnet_bs16_fixed.om" --input "bin_file_bs16" --output "res_data/bs16_out/" --outfmt TXT --device 1
```
### 5.3 脚本运行
运行如下脚本即可完成上述步骤，得到bs1和bs16各自的精度数据
```shell
bash test/eval_acc.sh
```
## 6. 精度对比
- 离线推理分类准确率
- pth模型推理分类准确率
- 开源分类准确率
- 准确率对比
### 6.1 离线推理分类准确率
后处理与准确率计算
batch_size为1时，运行pointnet_postprocess.py脚本与label比对，可以计算准确率并打印。

```shell
python3.7 pointnet_postprocess.py ./name2label.txt ./res_data/bs1_out batch_size=1
```
输出结果：
```
test accuracy: 0.9735
```
batch_size为16时，运行pointnet_postprocess.py脚本与label比对，可以计算准确率并打印。

```shell
python3.7 pointnet_postprocess.py ./name2label.txt ./res_data/bs16_out batch_size=16
```
输出结果：
```
test accuracy: 0.9728
```
经过对比bs1与bs16的om测试，本模型batch_size为1的精度与batch_size为16的精度有较小差距，原因在第5节已有说明。
### 6.2 pth模型推理分类准确率
运行eval.py脚本，得到pth推理的准确率：
```
test accuracy: 0.9742
```
### 6.3 开源分类准确率
github开源仓上得到的最好结果是：
```
test accuracy: 0.981
```
### 6.4 准确率对比
将得到的om离线模型推理准确率与该模型github代码仓上公布的精度对比，如下表所以，精度下降在1%范围内，故精度达标。
模型|Accuracy
-|-
github开源仓结果(官方)|98.1%
pth模型推理结果|97.42%
batch_size为1的om模型离线推理结果|97.35%
batch_size为16的om模型离线推理结果|97.28%

## 7. 性能评测
### 1. 310芯片性能数据
运行如下脚本可得到在310芯片上，每个batch_size的om模型对应的性能数据
```shell
bash test/perform_310.sh
```
### 2. gpu基准性能数据
运行如下脚本可得到在基准gpu上，每个batch_size的onnx模型对应的性能数据
```shell
bash test/perform_g.sh
```
### 3. 性能对比

Model|Batch Size|310(FPS/Card)|基准(FPS/Card)
-|-|-|-
PointNet|1|987|1787
PointNet|4|1058|2251
PointNet|8|1098|2367
PointNet|16|1102|2412
PointNet|32|1076|2380

