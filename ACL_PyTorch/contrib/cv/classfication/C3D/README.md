# C3D模型 Onnx端到端推理指导
## 1. 模型概述
### 1.1 论文地址
```shell
https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf
```
### 1.2 代码地址
- 进行推理之前先下载代码仓代码
```shell
https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/c3d/README.md
```
## 2. 环境说明
### 2.1 深度学习框架
```shell
CANN 5.0.2
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
在导出onnx文件前，请确保命令行当前路径为 `$MMACTION2/`。\
并用交附件中的`pytorch2onnx.py`替换`tools/pytorch2onnx.py`文件，运行如下脚本:
```shell
python tools/pytorch2onnx.py configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py checkpoints/C3D.pth --shape 1 10 3 16 112 112 --verify --softmax
```
参数说明：
- `--shape`: 模型输入张量的形状。对于C3D模型，输入形状为 `$batch $clip $channel $time $height $width`。
- `--verify`: 决定是否对导出模型进行验证，验证项包括是否可运行，数值是否正确等。如果没有被指定，它将被置为 `False`。
- `--show`: 决定是否打印导出模型的结构。如果没有被指定，它将被置为 `False`。
- `--softmax`: 是否在行为识别器末尾添加 Softmax。如果没有指定，将被置为 `False`。目前仅支持行为识别器，不支持时序动作检测器。
### 3.2 onnx转om模型
#### 5.2.1 设置环境变量
```shell
source env.sh
```
#### 3.2.2 使用atc将onnx模型转换为om模型
```shell
atc --framework=5 --model=C3D.onnx --output=C3D --input_format=ND --input_shape="image:1,10,3,16,112,112" --log=debug --soc_version=Ascend310 --auto_tune_mode=”RL,GA”
```
参数说明：

- `--model`: 输入的onnx模型路径。
- `--output`:输出的文件名。 
- `--input_format`: 输入形状的格式。
- `--input_shape`: 模型输入的形状。
- `--log`: 设置ATC模型转换过程中日志的级别
- `--soc_version`:soc版本
- `--auto_tune-mode`:是否开启auto-tune

## 4.数据预处理
### 4.1 数据集获取
用户可参考该数据集的 [官网](https://www.crcv.ucf.edu/research/data-sets/ucf101/)，以获取数据集相关的基本信息。
在数据集准备前，请确保命令行当前路径为 `$MMACTION2/tools/data/ucf101/`。
#### 4.1.1 下载数据集
下载视频文件，将下载好的视频数据放在`$mmaction2-master/data/videos/`下。
```shell
cd $tools/data/ucf101
bash download_videos.sh
```
#### 4.1.2 提取RGB原始帧
```shell
cd $tools/data/ucf101
bash extract_rgb_frames_opencv.sh
```
提取出来的原始帧放在`$mmaction2-master/data/ucf101/`路径下。
#### 4.1.3 检查目录结构
确认最终的目录结构是否是如下格式。
```
mmaction2-master
├── mmaction
├── tools
├── configs
├── data
│   ├── ucf101
│   │   ├── ucf101_{train,val}_split_{1,2,3}_rawframes.txt
│   │   ├── ucf101_{train,val}_split_{1,2,3}_videos.txt
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
python ./mmaction/datasets/rawframe_dataset.py ./configs/recognition/c3d/c3d_sports_sports1m_16x1x1_45e_ucf101_rgb.py --output_path ./data/prep_datasets
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
benchmark工具为华为自研的模型推理工具，支 持多种模型的离线推理，能够迅速统计出模型在 Ascend310上的性能，支持真实数据和
纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN V100R020C10 推理
benchmark工具用户指南 01 将获取的工具包并解压，将benchmark工具放在当前目录下
### 5.2 离线推理
#### 5.2.1 设置环境变量
```shell
source env.sh
```
#### 5.2.2 执行离线推理
运行如下脚本执行离线推理
```shell
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=C3D.om -input_text_path=./c3d_prep_bin.info -input_width=112 -input_height=112 -output_binary=False -useDvpp=False
```
输出结果保存在当前目录`result/dumpOutput_device0/`里；性能数据保存在`result/perf_vision_batchsize_1_device_0.txt`中

参数说明：
- `-model_type`:benchmark支持的模型类型，目前支持的有vision，nmt，widedeep，nlp，yolocaffe，bert，deepfm
- `-device_id`:运行在ascend 310的哪个device上，每张ascend 310卡有4个device
- `-batch_size`:om模型的batch大小，该值应与om模型的batch大小相同，否则报输入大小不一致的错误
- `-om_path`:om模型文件路径
- `-input_text_path`:包含数据集每个样本的路径与其相关信息的数据集信息文件路径
- `-input_height`:输入高度
- `-input_width`:输入宽度
- `-output_binary`:以预处理后的数据集为输入，benchmark工具推理om模型的输出数据保存为二进制还是txt，但对于输出是int64类型的节点时，指定输出为txt时会将float类型的小数转换为0而出错
- `-useDvpp`:是否使用aipp进行数据集预处理
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

运行之后会在result文件夹中生成`top1_acc.json`文件,得到精度数据为：
```shell
{"top1_acc": 0.818205874569992}
```
### 6.2 精度对比
|模型        |Top1精度 |
|:---------:|:------:|
|pth预训练模型|82.5    |
|om模型      |81.82   |

说明：

可以看到om模型的精度达到了pth模型精度的99.17%，精度达标，故不需要进行调试。
## 7. 性能对比
### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。 benchmark工具在整个数据集上推理获得性能数据:
1.batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt:
```shell
[e2e] throughputRate: 5.43018, latency: 695926
[data read] throughputRate: 5.6839, moduleLatency: 175.935
[preprocess] throughputRate: 5.52475, moduleLatency: 181.004
[infer] throughputRate: 5.43899, Interface throughputRate: 7.67659, moduleLatency: 181.881
[post] throughputRate: 5.43898, moduleLatency: 183.858
```
Interface throughputRate: 7.67659，7.67659x4=30.70636fps。即是batch1 310单卡吞吐率。
### 7.2 gpu性能数据
将C3D.onnx文件上传至208服务器，运行如下脚本：
```shell
trtexec --onnx=C3D.onnx --fp16 --shapes=image:1x10x3x16x112x112
```
得到
```shell
median=38.4012ms
```
计算batch 1 gpu单卡吞吐率：1000/(38.4012/1)=26.04085
### 7.3 性能对比
|设备  |单卡吞吐率|
|:---:|:------:|
|gpu  |26.04085|
|npu  |30.70636|

说明：
可以看到npu上的性能达到了gpu性能的1.179倍，性能达标，故不需要进行调试。
