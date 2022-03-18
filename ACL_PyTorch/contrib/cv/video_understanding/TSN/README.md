# 基于开源mmaction2预训练的TSN模型端到端推理指导

## 1 环境准备

1.安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

```shell
pip3.7 install -r requirements.txt  
```  

2.获取，修改与安装开源模型代码

```shell
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
git checkout 9ab8c2af52c561e5c789ccaf7b62f4b7679c103c
pip install -r requirements/build.txt
pip install -v -e .
cd ..
```


3.获取权重文件

需要获取tsn_r50_1x1x3_75e_ucf101_rgb/tsn_r50_1x1x3_75e_ucf101_rgb_20201023-d85ab600.pth文件，请参考文档。

4.数据集  

该模型使用UCF101的验证集进行测试，数据集下载步骤如下

```shell
cd ./mmaction2/tools/data/ucf101
bash download_annotations.sh
bash download_videos.sh
bash extract_rgb_frames_opencv.sh
bash generate_videos_filelist.sh
bash generate_rawframes_filelist.sh
```

（可选）本项目默认将数据集存放于/opt/npu/

```shell
cd ..
mv /ucf101 /opt/npu/
```

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)  
  将benchmark.x86_64或benchmark.aarch64放到当前目录  

6.使用msame工具推理
1.首先需要获取msame工具

```shell
git clone https://gitee.com/ascend/tools.git
```

2.而后安装msame工具

```shell
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub
cd ./tools/msame
./build.sh g++ ./
cd ../..
```

3.增加执行权限

```shell
chmod u+x ./tools/msame/out/msame
```
  
  
## 2 离线推理

310上执行，执行时使npu-smi info查看设备状态，确保device空闲

```
#OM model generation
bash test/pth2om.sh

#OM model inference
bash test/eval_acc_perf.sh --datasets_path=/root/datasets

#gpu inference
bash test/perf_gpu.sh
```

**评测结果：**

| 模型      | pth精度  | 310精度  | 性能基准    | 310性能    |
| :------: | :------: | :------: | :------:  | :------:  | 
| TSN bs1  | top1:83.03%| top1:82.84%|  22.63fps | 23.43fps | 
| TSN bs4 | top1:83.03%| top1:82.84%| 21.96fps | 24.41fps | 
| TSN bs8 | top1:83.03%| top1:82.84%| 22.18fps | 24.68fps |

