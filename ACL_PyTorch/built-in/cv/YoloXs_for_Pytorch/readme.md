###  YOLOX模型PyTorch离线推理指导

### 1. 环境准备

1. 安装依赖

```bash
pip3.7 install -r requirements.txt
```

若环境中已经安装了较多包，可手动补充安装所需要的包即可

1. 获取，修改与安装开源模型代码

```
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
git reset c9d128384cf0758723804c23ab7e042dbf3c967f --hard
pip3 install -v -e .
```

3. 在https://github.com/Megvii-BaseDetection/YOLOX 界面下载YOLOX-s对应的weights， 名称为yolox_s.pth。

4. 数据集

   使用coco2017数据集，请到https://cocodataset.org自行下载coco2017，放到/root/datasets目录(该目录与test/eval-acc-perf.sh脚本中的路径对应即可)，目录下包含val2017及annotations目录

5. [获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)

   将benchmark.x86_64或benchmark.aarch64放到当前工作目录

### 2. 离线推理

710上执行，执行时使npu-smi info查看设备状态，确保device空闲，设置环境变量后运行对应脚本即可，将modelzoo下载的本模型（YoloXs_for_Pytorch）下的文件及文件夹拷贝到YOLOX目录下

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd YOLOX
cp yolox_s.pth ./YOLOX
cp -r YoloXs_for_Pytorch/* ./YOLOX
bash test/pth2om.sh  
bash test/eval-acc-perf.sh --datasets_path=/root/datasets  
```

**评测结果：**

| 模型        | pth精度   | 710离线推理精度 | 性能基准  | 710性能 |
| ----------- | --------- | --------------- | --------- | ------- |
| yolox-s | map:40.5% | map:40.1%    |          | 950fps  |



