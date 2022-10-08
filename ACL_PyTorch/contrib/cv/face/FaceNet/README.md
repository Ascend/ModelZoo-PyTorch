#FACENET离线推理指导
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：?Facial Recognition**

**修改时间（Modified） ：2021.01.08**

**大小（Size）：110M**

**框架（Framework）：Pytorch 1.10.0**

**模型格式（Model Format）：pt**

**处理器（Processor）：昇腾310**

**描述（Description）：基于Pytorch框架的Facenet离线推理代码** 

<h2 id="概述.md">概述</h2>

FaceNet是一个通用人脸识别系统：采用深度卷积神经网络（CNN）学习将图像映射到欧式空间。空间距离直接和图片相似度相关：同一个人的不同图像在空间距离很小，不同人的图像在空间中有较大的距离，可以用于人脸验证、识别和聚类。在800万人，2亿多张样本集训练后，FaceNet在LFW数据集上测试的准确率达到了99.63%，在YouTube Faces DB数据集上，准确率为95.12%。

- 参考论文：

    [F. Schroff, D. Kalenichenko, J. Philbin. FaceNet: A Unified Embedding for Face Recognition and Clustering.” arXiv:1503.03832](https://https://arxiv.org/pdf/1503.03832.pdf) 

## 训练环境

* Pytorch 1.10.0
* Python 3.7.0

## 代码及路径解释

```
facenet_for_ACL
├── MTCNN_pth2onnx.py             MTCNN中的pt转成onnx文件
├── MTCNN_preprocess.py           MTCNN网络中实现数据预处理和推理
├── FaceNet_pth2onnx.py           FaceNet中的pt转成onnx文件
├── FaceNet_preprocess.py         生成FaceNet预处理所需要的bin文件
├── FaceNet_postprocess.py        FaceNet推理结果后处理，计算预测精度
├── gen_dataset_info.py           生成info文件
├── requirements.txt              项目依赖包清单
├── data                          数据集与网络结果输出		
│   └── ..                        
├── utils                         工具包文件		
│   └── ..                        
├── models                        om模型程序辅助文件
│   └── ..			            
├── weights                       存放pt/onnx/om文件		
│   └── ..		
├── test                          存放pth2om/test等脚本文件		
    └── ..			            
```


## 数据集
LFW (Labled Faces in the Wild)人脸数据集

测试集下载地址：链接: http://vis-www.cs.umass.edu/lfw/lfw.tgz

## 源码下载链接
```shell
git clone https://github.com/timesler/facenet-pytorch.git
```

## 优化ONNX模型
```shell
git clone https://gitee.com/Ronnie_zheng/MagicONNX  
cd MagicONNX && pip install .
```

## 添加mtcnn.py补丁
```shell
patch ./facenet-pytorch/models/mtcnn.py < ./facenet/models/mtcnn.patch
cp ./facenet-pytorch/models/mtcnn.py ./facenet/models/
```

## 模型文件
包括初始pt文件，导出的onnx文件，以及推理om文件  
链接：链接：https://pan.baidu.com/s/1hslY-6PZaqevSfZL3tWjwA  
提取码：1234 

## pt模型

生成onnx文件
```shell
python3 MTCNN_pth2onnx.py --model PNet --output_file ./weights/PNet_truncated.onnx
python3 MTCNN_pth2onnx.py --model RNet --output_file ./weights/RNet_truncated.onnx
python3 MTCNN_pth2onnx.py --model ONet --output_file ./weights/ONet_truncated.onnx
python3 FaceNet_pth2onnx.py --pretrain vggface2 --model ./weights/Inception_facenet_vggface2.pt --output_file ./weights/Inception_facenet_vggface2.onnx
```

优化onnx文件
```shell
python ./utils/fix_prelu.py ./weights/PNet_truncated.onnx ./weights/PNet_truncated_fix.onnx
python ./utils/fix_prelu.py ./weights/RNet_truncated.onnx ./weights/RNet_truncated_fix.onnx
python ./utils/fix_prelu.py ./weights/ONet_truncated.onnx ./weights/ONet_truncated_fix.onnx
python ./utils/fix_prelu.py ./weights/Inception_facenet_vggface2.onnx ./weights/Inception_facenet_vggface2_fix.onnx 
python ./utils/fix_clip.py ./weights/Inception_facenet_vggface2_fix.onnx ./weights/Inception_facenet_vggface2_fix.onnx 
```

## 生成om模型

使用ATC模型转换工具进行模型转换时可参考如下指令 atc.sh:
```shell
atc --framework=5 --model=./weights/PNet_truncated_fix.onnx --output=./weights/PNet_dynamic --input_format=NCHW --input_shape_range='images:[1~32,3,1-1500,1-1500]' --log=debug --soc_version=Ascend310 --log=error > atc1.log
atc --framework=5 --model=./weights/RNet_truncated_fix.onnx --output=./weights/RNet_dynamic --input_format=NCHW --input_shape_range='image:[1~2000,3,24,24]' --log=debug --soc_version=Ascend310 --log=error > atc1.log
atc --framework=5 --model=./weights/ONet_truncated_fix.onnx --output=./weights/ONet_dynamic --input_format=NCHW --input_shape_range='image:[1~1000,3,48,48]' --log=debug --soc_version=Ascend310 --log=error > atc1.log
atc --framework=5 --model=./weights/Inception_facenet_vggface2_fix.onnx --output=./weights/Inception_facenet_vggface2_bs1 --input_format=NCHW --input_shape="image:1,3,160,160" --soc_version=Ascend310 --log=error > atc1.log
```
具体参数使用方法请查看官方文档。

## MTCNN阶段的预处理与推理
该步骤完成MTCNN三个串行网络的预处理与推理后处理以生成cropped人脸图像  
batch_size 1
```shell
python3 MTCNN_preprocess.py --model Pnet --data_dir /home/ywj/facenet/data/lfw --batch_size 1
python3 MTCNN_preprocess.py --model Rnet --data_dir /home/ywj/facenet/data/lfw --batch_size 1
python3 MTCNN_preprocess.py --model Onet --data_dir /home/ywj/facenet/data/lfw --batch_size 1
```
batch_size 16
```shell
python3 MTCNN_preprocess.py --model Pnet --data_dir /home/ywj/facenet/data/lfw --batch_size 1
python3 MTCNN_preprocess.py --model Rnet --data_dir /home/ywj/facenet/data/lfw --batch_size 1
python3 MTCNN_preprocess.py --model Onet --data_dir /home/ywj/facenet/data/lfw --batch_size 1
```
## 将MTCNN生成的cropped图像预处理转成bin文件

将cropped的人脸图像转为bin文件
batch_size 1
```shell
python3 FaceNet_preprocess.py --crop_dir ./data/lfw_split_om_cropped_1 --save_dir ./data/input/Facenet_1 --batch_size 1
```
batch_size 16
```shell
python3 FaceNet_preprocess.py --crop_dir ./data/lfw_split_om_cropped_16 --save_dir ./data/input/Facenet_16 --batch_size 16
```

## 使用msame工具进行离线推理 以下均采用vggface2为案例
batch_size 1
```shell
./msame --model ./weights/Inception_facenet_vggface2_bs1.om --input ./data/input/Facenet_1/xb_results --output ./data/output
```
batch_size 16
```shell
./msame --model ./weights/Inception_facenet_vggface2_bs16.om --input ./data/input/Facenet_16/xb_results --output ./data/output
```
参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。


## 使用推理得到的bin文件进行精度测试
batch_size 1
```shell
python3 ./utils/batch_utils.py --batch_size 1 --data_root_path ./data/output/Facenet_vggface2_1 --save_root_path ./data/output/Facenet_vggface2_1_2
python3 FaceNet_postprocess.py  --ONet_output_dir ./data/output/split_bs1/onet.json --test_dir ./data/output/Facenet_vggface2_1_2 --crop_dir ./data/lfw_split_om_cropped_1
```
batch_size 16
```shell
python3 ./utils/batch_utils.py --batch_size 16 --data_root_path ./data/output/Facenet_vggface2_16 --save_root_path ./data/output/Facenet_vggface2_16_2
python3 FaceNet_postprocess.py  --ONet_output_dir ./data/output/split_bs16/onet.json --test_dir ./data/output/Facenet_vggface2_16_2 --crop_dir ./data/lfw_split_om_cropped_16
```

## 精度

* Ascend310推理精度：

| Model name | LFW accuracy  | Training dataset    | 
| :--------: | ---------- | ------ |
|   20180402-114759 (107MB)  | 0.994     | VGGFace2(bs1)   |
|   20180402-114759 (107MB)  | 0.994     | VGGFace2(bs16)   | 

* GPU推理精度：

| Model name | LFW accuracy  | Training dataset    | 
| :--------: | ---------- | ------ |
|   Inception_facenet_vggface2_bs1.om  | 0.994     | VGGFace2(bs1)   |
|   Inception_facenet_vggface2_bs16.om  | 0.994     | VGGFace2(bs16)   | 

## 推理性能：
* Ascend310推理性能：

| Model name | Input_shape | FPS  | AvgLatency  | 
| :--------: | ----------  | ---- |   ------    |
|   PNet_dynamic.om  | 1×3×1500×1500 |  164.04   | 24.384ms |
|   PNet_dynamic.om  | 4×3×1500×1500 |  189.66   | 21.582ms |
|   PNet_dynamic.om  | 8×3×1500×1500 |  172.45   | 23.194ms |
|   RNet_dynamic.om  | 100×3×24×24   |  446.48   | 8.9590ms |
|   RNet_dynamic.om  | 800×3×24×24   |  1229.5   | 3.2534ms |
|   RNet_dynamic.om  | 1600×3×24×24  |  1401.9   | 2.8533ms |
|   ONet_dynamic.om  | 50×3×48×48    |  335.99   | 11.905ms |
|   ONet_dynamic.om  | 400×3×48×48   |  872.35   | 4.5853ms |
|   ONet_dynamic.om  | 800×3×48×48   |  947.44   | 4.2219ms |
|   Inception_facenet_vggface2_bs1.om       | 1×3×160×160  | 1693.7   | 2.3714ms    | 
|   Inception_facenet_vggface2_bs4.om       | 4×3×160×160  | 3034.6   | 1.4485ms    | 
|   Inception_facenet_vggface2_bs8.om       | 8×3×160×160  | 4553.5   | 0.9154ms    | 
|   Inception_facenet_vggface2_bs16.om      | 16×3×160×160 | 5336.4   | 0.7500ms    |
|   Inception_facenet_vggface2_bs32.om      | 32×3×160×160 | 3850.5   | 1.0391ms    |
  
* Onnx性能 

| Model name | Input_shape | FPS  | AvgLatency | 
| :--------: | ----------- | ---- | ------ |
| PNet_sim_1_3_1500_1500.onnx | 1×3×1500×1500 | 281.281 | 3.5552ms |
| PNet_sim_4_3_1500_1500.onnx | 4×3×1500×1500 | 293.238 | 3.4102ms |
| PNet_sim_8_3_1500_1500.onnx | 8×3×1500×1500 | 288.747 | 3.4633ms |
| RNet_sim_100_3_24_24.onnx   | 100×3×24×24   | 5288.21 | 0.1881ms |
| RNet_sim_800_3_24_24.onnx   | 800×3×24×24   | 7873.63 | 0.1270ms |
| RNet_sim_1600_3_24_24.onnx  | 1600×3×24×24  | 8367.58 | 0.1195ms |
| ONet_sim_50_3_48_48.onnx    | 50×3×48×48    | 2643.38 | 0.3783ms |
| ONet_sim_400_3_48_48.onnx   | 400×3×48×48   | 3286.14 | 0.3043ms |
| ONet_sim_800_3_48_48.onnx   | 800×3×48×48   | 3438.91 | 0.2908ms |
| Inception_vggface2_sim_1_3_160_160.onnx |  1×3×160×160  | 797.067 | 1.2546ms |
| Inception_vggface2_sim_4_3_160_160.onnx |  4×3×160×160  | 2473.96 | 0.4042ms |
| Inception_vggface2_sim_8_3_160_160.onnx |  8×3×160×160  | 3724.76 | 0.2685ms |
| Inception_vggface2_sim_16_3_160_160.onnx|  16×3×160×160 | 4727.60 | 0.2116ms |
| Inception_vggface2_sim_32_3_160_160.onnx|  32×3×160×160 | 5583.47 | 0.1791ms |