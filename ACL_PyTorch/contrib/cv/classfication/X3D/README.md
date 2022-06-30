# X3D模型ONNX离线推理指导

## 1 环境准备

1.源码获取：根据[ATC X3D(FP16)-昇腾社区 (hiascend.com)](https://www.hiascend.com/zh/software/modelzoo/detail/1/1f626937bdac487087bd2debb16c6d7d)中快速上手->获取源码来下载源码。然后，需要安装必要的依赖，测试的base环境可能已经安装其中的一些不同版本的库了，故手动测试时需要使用conda创建一个虚拟环境，并安装这些基础依赖。

| 依赖名称    |  版本   |
| ----------- | :-----: |
| onnx        |  1.9.0  |
| torch       |  1.8.0  |
| torchvision |  0.9.0  |
| numpy       | 1.20.3  |
| pillow      |  8.2.0  |
| CANN        | 5.1.RC1 |

```
conda create -n 环境名 python=3.8
```

```
pip3 install -r requirements.txt
```
2.获取，修改与安装开源模型代码
```
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip3 install -e detectron2_repo

git clone https://github.com/facebookresearch/SlowFast -b master
cd SlowFast

git reset 9839d1318c0ae17bd82c6a121e5640aebc67f126 --hard
mv ../x3d.patch ./
patch -p1 < x3d.patch
python3.7 setup.py build develop

cd ..

```
3.[获取权重文件 x3d_s.pyth](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)

    将权重文件x3d_s.pyth放在当前目录。

4.获取数据集Kinetics-400

脚本下载:    
获取验证集列表文件[val_link.list](https://ai-rank.bj.bcebos.com/Kinetics400/val_link.list)与验证集标签文件[val.list](https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val.list)，并将val.list重命名为test.csv
下载验证集：

```
download.sh:
file=$1

while read line 
do
    wget "$line"
done <$file

bash download.sh val_link.list
```
将下载的val_part1,val_part2,val_part3里的400个文件夹放到/root/datasets/Knetics-400/val，将test.csv放到X3D/Kinetics-400。

5.获取 [msame工具](https://gitee.com/ascend/tools/tree/master/msame)
和
[benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)

将msame和benchmark.x86_64（或benchmark.aarch64）放到当前目录


## 2 离线推理

310上执行，执行时使npu-smi info查看设备状态，确保device空闲，并执行以下命令

310P上执行和310上执行一样，只需要把--soc_version参数改为Ascend310P，然后执行以下命令

```
bash test/pth2om.sh  
bash test/eval_acc_pref.sh --datasets_path=X3D/Kinetics-400
```
备注：存在fp16算子溢出，精度不达标，因此atc模型转换需要添加--precision_mode allow_mix_precision

**评测结果：**

| 模型     |  官网pth精度   | 310精度 | 基准性能| 310性能 |
| ----    | ------------------------------------------------------------ | --------------- | -------- | ------- |
| X3d bs1 |  [Top1:73.1%](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)  | Top1:73.75%   Top5:90.25% | 95.07fps   | 173.66fps |
| X3d bs16|  [Top1:73.1%](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)  | Top1:73.75%   Top5:90.25% | 103.82fps  | 125.29fps |

| 模型     | 710精度                   | 710性能   |
| -------- | ------------------------- | --------- |
| X3d bs1  | Top1:73.75%   Top5:90.25% | 313.10fps |
| X3d bs16 | Top1:73.75%   Top5:90.25% | 399.68fps |

