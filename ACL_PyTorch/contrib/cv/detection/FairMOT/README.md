# FairMOT模型离线推理指导
# 1.环境准备
## 1.安装依赖
~~~
pip3.7 install -r requirements.txt
~~~


## 2.安装DCN以及修改DCN代码
~~~
git clone -b pytorch_1.5 https://github.com/ifzhang/DCNv2.git
cd DCNv2
python3.7 setup.py build develop
git reset 9f4254babcd162a809d165fa2430a780d14761f4 --hard
patch -p1 < ../dcnv2.diff  
cd ..
~~~

## 3.下载并修改开源模型代码
~~~
git clone -b master https://github.com/ifzhang/FairMOT.git
cd FairMOT
git reset 2f36e7ebf640313a422cb7f07f93dc53df9b8d12 --hard
patch -p1 < ../fairmot.diff
cd ..
~~~

## 4.下载权重文件
[fairmot_dla34.pth](https://drive.google.com/file/d/1iqRQjsG9BawIl8SlFomMg5iwkb6nqSpi/view?usp=sharing) 
## 5.准备数据集
~~~
mkdir dataset
cd dataset
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip
cd MOT17
mkdir images
mv train/ images/
mv test/ images/
cd ../..
python3.7 FairMOT/src/gen_labels_16.py
~~~
## 6.[获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)  
将benchmark.x86_64或benchmark.aarch64放到当前目录

# 2.离线推理
310上执行，执行时使npu-smi info查看设备状态，确保device空闲  
~~~
bash test/pth2om.sh
bash test/eval_acc_perf.sh
~~~

 **评测结果：**   
| 模型      | pth精度  | 310离线推理精度  | 基准性能    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| FairMOT bs1  | MOTA : 83.8  | MOTA:83.7 |  12.65fps |  8.01fps|
|FariMOT bs8 |-|MOTA:83.7 |-|8.17fps|

备注：包含dcn自定义算子，使用在线推理评测，在线推理只支持bs1。由于内存限制，只支持到bs8的离线推理


















