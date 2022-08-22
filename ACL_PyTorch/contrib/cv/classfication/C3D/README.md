# C3D模型 Onnx端到端推理指导
## 1. 模型概述
### 1.1 论文地址
```shell
https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf
```
### 1.2 代码地址
- 进行推理之前先下载代码仓代码
- branch: master commit id : 6d6685632f28344e98cf34a14d1226cd6c008391
```shell
https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/c3d/README.md
```
## 2. 环境说明
### 2.1 深度学习框架
```shell
CANN 5.1.RC1
torch==1.6.0
torchvision==0.7.0
onnx==1.10.2
onnxruntime==1.9.0
mmcv==1.3.17
```
### 2.2 python第三方库
```shell
opencv-python==4.5.4.58
numpy==1.21.2
pillow==8.4.0
```
## 3. 模型转换
### 3.1 pth转onnx模型
#### 3.1.1 下载pth权重文件
pth文件使用310训练得到的权重文件，下载后放在`$mmaction2-master/checkpoints/`下
#### 3.1.2 执行pth2onnx.py脚本，生成onnx模型文件
在导出onnx文件前，请确保命令行当前路径为 `$mmaction2-master/`。\
并用交附件中的`pytorch2onnx.py`替换`tools/deployment/pytorch2onnx.py`文件，运行如下脚本:
```shell
python tools/deployment/pytorch2onnx.py configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py checkpoints/C3D.pth --shape 1 10 3 16 112 112 --verify --softmax --output-file=C3D.onnx
```
参数说明：
- `--shape`: 模型输入张量的形状。对于C3D模型，输入形状为 `$batch $clip $channel $time $height $width`。
- `--verify`: 决定是否对导出模型进行验证，验证项包括是否可运行，数值是否正确等。如果没有被指定，它将被置为 `False`。
- `--show`: 决定是否打印导出模型的结构。如果没有被指定，它将被置为 `False`。
- `--softmax`: 是否在行为识别器末尾添加 Softmax。如果没有指定，将被置为 `False`。目前仅支持行为识别器，不支持时序动作检测器。
- `--output-file`: 如果没有指定，将被置为 `tmp.onnx`。
### 3.2 onnx转om模型
#### 5.2.1 设置环境变量
```shell
source set_env.sh
```
#### 3.2.2 使用atc将onnx模型转换为om模型
```shell
atc --framework=5 --model=C3D.onnx --output=C3D --input_format=ND --input_shape="image:1,10,3,16,112,112" --log=debug --soc_version=${chip_name}
```
参数说明：

- `--model`: 输入的onnx模型路径。
- `--output`:输出的文件名。 
- `--input_format`: 输入形状的格式。
- `--input_shape`: 模型输入的形状。
- `--log`: 设置ATC模型转换过程中日志的级别。
- `--soc_version`: ${chip_name}可通过`npu-smi info`指令查看 。

## 4.数据预处理
### 4.1 数据集获取
用户可参考该数据集的 [官网](https://www.crcv.ucf.edu/research/data-sets/ucf101/)，以获取数据集相关的基本信息。
在数据集准备前，请确保命令行当前路径为 `$mmaction2-master/tools/data/ucf101/`。
#### 4.1.1 下载数据集

下载标注文件；下载视频文件，将下载好的视频数据放在`$mmaction2-master/data/videos/`下。

```shell
cd $tools/data/ucf101
bash download_annotations.sh
bash download_videos.sh
```
#### 4.1.2 提取RGB原始帧
```shell
cd $tools/data/ucf101
bash extract_rgb_frames_opencv.sh
```
提取出来的原始帧放在`$mmaction2-master/data/ucf101/`路径下。
#### 4.1.3 生成标注文件
```shell
cd $tools/data/ucf101
bash generate_rawframes_filelist.sh
```
#### 4.1.4 检查标注文件

将交付件中的`check_rawframes_filelist.sh`放在`$tools/data/ucf101`目录下，并在该目录下运行，剔除问题样本（数据集中的v_PommelHorse_g05序列会生成大小不符合的帧样本）。

```shell
cd $tools/data/ucf101
bash check_rawframes_filelist.sh
```

#### 4.1.5 检查目录结构

确认最终的目录结构是否是如下格式。
```
mmaction2-master
├── mmaction
├── tools
├── configs
├── data
│   ├── ucf101
│   │   ├── ucf101_{train,val}_split_{1,2,3}_rawframes.txt
│   │   ├── ucf101_{train,val}_split_{1,2,3}_videos.txt #可以没有
│   │   ├── annotations
│   │   ├── videos
│   │   │   ├── ApplyEyeMakeup
│   │   │   │   ├── v_ApplyEyeMakeup_g01_c01.avi

│   │   │   ├── YoYo
│   │   │   │   ├── v_YoYo_g25_c05.avi
│   │   ├── rawframes
│   │   │   ├── ApplyEyeMakeup
│   │   │   │   ├── v_ApplyEyeMakeup_g01_c01
│   │   │   │   │   ├── img_00001.jpg
│   │   │   │   │   ├── img_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_x_00001.jpg
│   │   │   │   │   ├── flow_x_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_y_00001.jpg
│   │   │   │   │   ├── flow_y_00002.jpg
│   │   │   ├── ...
│   │   │   ├── YoYo
│   │   │   │   ├── v_YoYo_g01_c01
│   │   │   │   ├── ...
│   │   │   │   ├── v_YoYo_g25_c05
```
### 4.2 数据集预处理
将交附件中的`rawframe_dataset.py`替换代码仓中`$mmaction2-master/mmaction/datasets`路径下的`rawframe_dataset.py`文件。\
确保当前工作目录为：`$mmaction2-master/`
运行：

```shell
python ./mmaction/datasets/rawframe_dataset.py ./configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py --output_path ./data/prep_datasets
```
将原始帧（rawframes）处理为bin文件。\
注意：在处理之前，需要提前在对应路径下创建好prep_datasets文件夹。\
参数说明：

- 参数1：config文件的路径
- 参数2：输出文件夹的位置
### 4.3 生成数据集信息文件
将处理好的数据，生成对应的info文件，作为benchmark工具推理的输入。
参考代码：
```shell
python get_info.py bin ./prep_datasets ./c3d_prep_bin.info 112 112
```
参数说明：
- `参数1`：bin文件所在的文件夹的路径
- `参数2`：为输出的info文件的名称
- `参数3、4`：分别表示每张图片的宽和高
## 5. 离线推理
### 5.1 benchmark工具概述
benchmark工具为华为自研的模型推理工具，支 持多种模型的离线推理，能够迅速统计出模型在芯片上的性能，支持真实数据和
纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN V100R020C10 推理
benchmark工具用户指南 01 将获取的工具包并解压，将benchmark工具放在当前目录下

### 5.2 离线推理
#### 5.2.1 设置环境变量
```shell
source set_env.sh
```
#### 5.2.2 执行离线推理
- 使用ais-infer工具进行推理。

  python3 ais_infer.py -–model ./C3D_16.om --input=./prep_ datasets/ --output ./result –outfmt TXT --batchsize=16 --infer_queue_count 1

  \-  参数说明：

   

    \-  model：需要进行推理的om模型。

    \-  input：模型需要的输入，支持bin文件和目录，若不加该参数，会自动生成都为0的数据。

  \-  output：推理结果输出路径。默认会建立日期+时间的子文件夹保存输出结果 如果指定output_dirname 将保存到output_dirname的子文件夹下。。

  \-  outfmt：输出数据的格式，默认”BIN“，可取值“NPY”、“BIN”、“TXT”。

  \-  batchsize：模型batch size 默认为1 。当前推理模块根据模型输入和文件输出自动进行组batch。参数传递的batchszie有且只用于结果吞吐率计算。请务必注意需要传入该值，以获取计算正确的吞吐率。

  \-  infer_queue_count：推理队列的数据最大数 可选参数，默认20。如果推理输入输出数据内存比较大，可能超过内存容量时，需要调小该值。

   

    推理后的输出默认在当前目录result下。

    \>**说明：** 

    \>执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer。
## 6. 精度对比
### 6.1 离线推理Top1精度
将result文件夹、标注文件和精度统计代码放在同一个文件夹内，运行如下脚本评测精度：
```shell
python C3D_postprocess.py ./result/dumpOutput_device0 ./mmaction2-master/data/ucf101/ucf101_val_split_1_rawframes.txt ./result/top1_acc.json
```
参数说明：
- `参数1`:离线推理得到的结果文件夹所在的路径
- `参数2`:标注文件所在的路径
- `参数3`:输出的json文件保存路径，json文件中保存了精度数据

运行之后会在result文件夹中生成`top1_acc.json`文件，在310上得到精度数据为：
```shell
{"top1_acc": 0.8189997353797301}
```
在310P上得到精度数据为：
```shell
{'top1_acc': 0.8189997353797301}
```
### 6.2 精度对比
|       模型        | Top1精度 |
| :---------------: | :------: |
| pth预训练模型(T4) |  82.24   |
|    om模型(310)    |  81.89   |
|   om模型(310P)    |  81.89   |

说明：可以看到om模型在310上精度达到了pth模型精度的99.57%，可以看到om模型在310P上的精度达到了pth模型精度的99.57%，精度达标，故不需要进行调试。\
备注：源码仓Top1精度为83.27。

## 7. 性能对比
### 7.1 npu性能数据

1.batch1的性能:

```shell
[e2e] throughputRate: 4.28779, latency: 881339
[data read] throughputRate: 4.5526, moduleLatency: 219.655
[preprocess] throughputRate: 4.39146, moduleLatency: 227.715
[inference] throughputRate: 4.29808, Interface throughputRate: 7.59556, moduleLatency: 229.201
[postprocess] throughputRate: 4.2992, moduleLatency: 232.602
fps=7.59556*4 = 30.38224 
```
Interface throughputRate: 7.59556，7.59556x4=30.38224fps。即是batch1 310单卡吞吐率。
```shell
[INFO] load model /home/ys/C3D/C3D_1.om success
[INFO] create model description success
[INFO] output path:/home/ys/C3D/ais_result/2022_08_08-03_56_49
[INFO] warm up 5 times done
[INFO] get filesperbatch files0 size:24084480 tensor0size:24084480 filesperbatch:1 runcount:3779
Inference Processing task: 100%|███████████████████████████████| 3779/3779 [08:36<00:00,  7.31it/s]
[INFO] -----------------Performance Summary------------------
[INFO] H2D_latency (ms): min = 3.576040267944336, max = 25.34198760986328, mean = 4.1277097564236955, median = 3.843069076538086, percentile(99%) = 14.916601181030234
[INFO] NPU_compute_time (ms): min = 18.80500030517578, max = 33.78099822998047, mean = 19.04405264540241, median = 18.972999572753906, percentile(99%) = 19.706099739074705
[INFO] D2H_latency (ms): min = 0.0286102294921875, max = 13.07225227355957, mean = 0.10634862925521152, median = 0.05269050598144531, percentile(99%) = 0.43358325958251764
[INFO] throughput 1000*batchsize(1)/NPU_compute_time.mean(19.04405264540241): 52.50983173696586
[INFO] ------------------------------------------------------
[INFO] unload model success, model Id is 1
DestroyDevices begindestory device:0
aclrtDestroyContext successfully!
DestroyDevices successfully
```
Interface throughputRate: 52.50983fps。即是batch1 310P单卡吞吐率。
### 7.2 gpu性能数据
将C3D.onnx文件上传至208服务器，运行如下脚本：
```shell
trtexec --onnx=C3D.onnx --fp16 --shapes=image:1x10x3x16x112x112
```
得到
```shell
mean = 26.10020 ms
```
计算batch 1 gpu单卡吞吐率：1000/(37.3741/1)=26.75650
### 7.3 性能对比
|   设备    | 单卡吞吐率 |
| :-------: | :--------: |
|    gpu    |  26.10020  |
| npu(310)  |  30.38224  |
| npu(310P) |  52.50983  |

说明：可以看到npu(310)上的性能达到了gpu性能的1.164倍，npu(310P)上的性能达到了gpu性能的2.011倍，性能达标，故不需要进行调试。           
