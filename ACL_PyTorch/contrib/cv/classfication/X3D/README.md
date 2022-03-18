# X3D模型PyTorch离线推理指导

## 1 环境准备

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

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
patch -p1 < x3d.patch
python3.7 setup.py build develop

cd ..

``` 
3.[获取权重文件 x3d_s.pyth](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)

    将权重文件x3d_s.pyth放在当前目录。

4.获取数据集Knetics-400

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

download.sh val_link.list
```
将下载的val_part1,val_part2,val_part3里的400个文件夹放到/root/datasets/Knetics-400/val，将test.csv放到/root/datasets/Knetics-400。

5.获取 [msame工具](https://gitee.com/ascend/tools/tree/master/msame)
和
[benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)

将msame和benchmark.x86_64（或benchmark.aarch64）放到当前目录


## 2 离线推理

310上执行，执行时使npu-smi info查看设备状态，确保device空闲

```
bash test/pth2om.sh  
bash test/evl_acc_pref.sh --datasets_path=/root/datasets/Knetics-400
```
备注：存在fp16算子溢出，精度不达标，因此atc模型转换需要添加--precision_mode allow_mix_precision

**评测结果：**

| 模型     |  官网pth精度   | 310精度 | 基准性能| 310性能 |
| ----    | ------------------------------------------------------------ | --------------- | -------- | ------- |
| X3d bs1 |  [Top1:73.1%](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)  | Top1:72.86%   Top5:89.45%   | 95.07fps   | 158.57fps |
| X3d bs16|  [Top1:73.1%](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)  | Top1:72.86%   Top5:89.45%   | 103.82fps  | 115.34fps  |