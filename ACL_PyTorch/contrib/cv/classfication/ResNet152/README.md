# ResNet152 Onnxģ�Ͷ˵�������ָ��
-   [1 ģ�͸���](#1-ģ�͸���)
	-   [1.1 ���ĵ�ַ](#11-���ĵ�ַ)
	-   [1.2 �����ַ](#12-�����ַ)
-   [2 ����˵��](#2-����˵��)
	-   [2.1 ���ѧϰ���](#21-���ѧϰ���)
	-   [2.2 python��������](#22-python��������)
-   [3 ģ��ת��](#3-ģ��ת��)
	-   [3.1 pthתonnxģ��](#31-pthתonnxģ��)
	-   [3.2 onnxתomģ��](#32-onnxתomģ��)
-   [4 ���ݼ�Ԥ����](#4-���ݼ�Ԥ����)
	-   [4.1 ���ݼ���ȡ](#41-���ݼ���ȡ)
	-   [4.2 ���ݼ�Ԥ����](#42-���ݼ�Ԥ����)
	-   [4.3 �������ݼ���Ϣ�ļ�](#43-�������ݼ���Ϣ�ļ�)
-   [5 ��������](#5-��������)
	-   [5.1 benchmark���߸���](#51-benchmark���߸���)
	-   [5.2 ��������](#52-��������)
-   [6 ���ȶԱ�](#6-���ȶԱ�)
	-   [6.1 ��������TopN����ͳ��](#61-��������TopN����ͳ��)
	-   [6.2 ��ԴTopN����](#62-��ԴTopN����)
	-   [6.3 ���ȶԱ�](#63-���ȶԱ�)
-   [7 ���ܶԱ�](#7-���ܶԱ�)
	-   [7.1 npu��������](#71-npu��������)
	-   [7.2 T4��������](#72-T4��������)
	-   [7.3 ���ܶԱ�](#73-���ܶԱ�)



## 1 ģ�͸���

-   **[���ĵ�ַ](#11-���ĵ�ַ)**  

-   **[�����ַ](#12-�����ַ)**  

### 1.1 ���ĵ�ַ
[ResNet152����](https://arxiv.org/pdf/1512.03385.pdf)  

### 1.2 �����ַ
[ResNet152����](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)  
branch:master
commit_id:02e6da5189b22870c549470485d68fff23d511bf
          

## 2 ����˵��

-   **[���ѧϰ���](#21-���ѧϰ���)**  

-   **[python��������](#22-python��������)**  

### 2.1 ���ѧϰ���
```
CANN 5.0.1

pytorch >= 1.5.0
torchvision >= 0.6.0
onnx >= 1.7.0
```

### 2.2 python��������

```
numpy == 1.18.5
Pillow == 7.2.0
opencv-python == 4.5.1.52
```

**˵����** 
>   X86�ܹ���pytorch��torchvision��onnx����ͨ���ٷ�����whl����װ����������ͨ��pip3.7 install ���� ��װ
>
>   Arm�ܹ���pytorch��torchvision��onnx����ͨ��Դ����밲װ����������ͨ��pip3.7 install ���� ��װ

## 3 ģ��ת��

-   **[pthתonnxģ��](#31-pthתonnxģ��)**  

-   **[onnxתomģ��](#32-onnxתomģ��)**  

### 3.1 pthתonnxģ��

1.����pthȨ���ļ�  
[ResNet152Ԥѵ��pthȨ���ļ�](https://download.pytorch.org/models/resnet152-b121ed2d.pth)  
```
wget https://download.pytorch.org/models/resnet152-b121ed2d.pth
```
�ļ�MD5sum��d3ddb494358a7e95e49187829ec97395

2.ResNet152ģ�ʹ�����torchvision���װtorchvision��arm����Դ�밲װ���ο�torchvision����������װ���̱�����ٶȽ��
```
git clone https://github.com/pytorch/vision
cd vision
python3.7 setup.py install
cd ..
```
3.��дpth2onnx�ű�resnet152_pth2onnx.py

 **˵����**  
>ע��ĿǰATC֧�ֵ�onnx���Ӱ汾Ϊ11

4.ִ��pth2onnx�ű�������onnxģ���ļ�
```
python3.7 resnet152_pth2onnx.py ./resnet152-f37072fd.pth resnet152.onnx
```

 **ģ��ת��Ҫ�㣺**  
>��ģ��ת��Ϊonnx����Ҫ�޸Ŀ�Դ����ִ��룬�ʲ���Ҫ����˵��

### 3.2 onnxתomģ��

1.���û�������
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.ʹ��atc��onnxģ��ת��Ϊomģ���ļ�������ʹ�÷������Բο�[CANN V100R020C10 ������������ָ�� (����) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)
```
atc --framework=5 --model=./resnet152.onnx --output=resnet152_bs32 --input_format=NCHW --input_shape="image:32,3,224,224" --log=debug --soc_version=Ascend310

```

## 4 ���ݼ�Ԥ����

-   **[���ݼ���ȡ](#41-���ݼ���ȡ)**  

-   **[���ݼ�Ԥ����](#42-���ݼ�Ԥ����)**  

-   **[�������ݼ���Ϣ�ļ�](#43-�������ݼ���Ϣ�ļ�)**  

### 4.1 ���ݼ���ȡ
��ģ��ʹ��[ImageNet����](http://www.image-net.org)��5������֤�����в��ԣ�ͼƬ���ǩ�ֱ�����/root/datasets/imagenet/val��/root/datasets/imagenet/val_label.txt��

### 4.2 ���ݼ�Ԥ����
1.Ԥ�����ű�imagenet_torch_preprocess.py

2.ִ��Ԥ�����ű����������ݼ�Ԥ�������bin�ļ�
```
python3.7 imagenet_torch_preprocess.py resnet /root/datasets/imagenet/val ./prep_dataset
```
### 4.3 �������ݼ���Ϣ�ļ�
1.�������ݼ���Ϣ�ļ��ű�gen_dataset_info.py

2.ִ���������ݼ���Ϣ�ű����������ݼ���Ϣ�ļ�
```
python3.7 gen_dataset_info.py bin ./prep_dataset ./resnet152_prep_bin.info 224 224
```
��һ������Ϊģ����������ͣ��ڶ�������Ϊ���ɵ�bin�ļ�·����������Ϊ�����info�ļ�������Ϊ������Ϣ
## 5 ��������

-   **[benchmark���߸���](#51-benchmark���߸���)**  

-   **[��������](#52-��������)**  

### 5.1 benchmark���߸���

benchmark����Ϊ��Ϊ���е�ģ���������ߣ�֧�ֶ���ģ�͵������������ܹ�Ѹ��ͳ�Ƴ�ģ����Ascend310�ϵ����ܣ�֧����ʵ���ݺʹ���������ģʽ����Ϻ����ű�������ʵ�����ģ�͵Ķ˵��˹��̣���ȡ���߼�ʹ�÷������Բο�[CANN V100R020C10 ����benchmark�����û�ָ�� 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)
### 5.2 ��������
1.���û�������
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.ִ����������
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=resnet152_bs1.om -input_text_path=./resnet152_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False

./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=resnet152_bs16.om -input_text_path=./resnet152_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False

./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=4 -om_path=resnet152_bs4.om -input_text_path=./resnet152_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False

```
������Ĭ�ϱ����ڵ�ǰĿ¼result/dumpOutput_device{0}��ģ��ֻ��һ����Ϊclass�������shapeΪbs * 1000����������ΪFP32����Ӧ1000�������Ԥ������ÿ�������Ӧ�������Ӧһ��_x.bin�ļ���

## 6 ���ȶԱ�

-   **[��������TopN����](#61-��������TopN����)**  
-   **[��ԴTopN����](#62-��ԴTopN����)**  
-   **[���ȶԱ�](#63-���ȶԱ�)**  

### 6.1 ��������TopN����ͳ��

����ͳ��TopN����

����imagenet_acc_eval.py�ű����������label�ȶԣ����Ի��Accuracy Top5���ݣ����������result.json�С�
```
python3.7 imagenet_acc_eval.py result/dumpOutput_device0/ /root/datasets/imagenet/val_label.txt ./ result.json
```
��һ��Ϊbenchmark���Ŀ¼���ڶ���Ϊ���ݼ����ױ�ǩ���������������ļ��ı���Ŀ¼�����ĸ������ɵ��ļ�����  
�鿴��������
```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "78.31%"}, {"key": "Top2 accuracy", "value": "87.83%"}, {"key": "Top3 accuracy", "value": "91.25%"}, {"key": "Top4 accuracy", "value": "92.97%"}, {"key": "Top5 accuracy", "value": "94.05%"}]}
```
������bs1��bs16��om���ԣ���ģ��batch1�ľ�����batch16�ľ���û�в�𣬾������ݾ�����

### 6.2 ��ԴTopN����
[torchvision��������](https://pytorch.org/vision/stable/models.html)
```
Model        Acc@1     Acc@5
resnet152    78.312	   94.046
```
### 6.3 ���ȶԱ�
���õ���om����ģ������TopN�������ģ��github������Ϲ����ľ��ȶԱȣ������½���1%��Χ֮�ڣ��ʾ��ȴ�ꡣ  
 **���ȵ��ԣ�**  
>û���������Ȳ��������⣬�ʲ���Ҫ���о��ȵ���

## 7 ���ܶԱ�

-   **[npu��������](#71-npu��������)**  
-   **[T4��������](#72-T4��������)**  
-   **[���ܶԱ�](#73-���ܶԱ�)**  

### 7.1 npu��������
benchmark�������������ݼ�������ʱҲ��ͳ���������ݣ����������������ݼ������������ô��������ô���������ڼ���Ҫȷ����ռdevice��ʹ��npu-smi info���Բ鿴device�Ƿ���С�Ҳ����ʹ��benchmark���������ܲ���������ݣ������������������ģ�����ݷֲ������������ܲ����Щģ���������ݿ��ܲ�̫׼��benchmark���������ܲ����ܽ�Ϊ���ٻ�ȡ��ŵ����������Ա�����Ż�ʹ�ã��ɳ���ȷ��benchmark�������������ݼ�������ʱ����deviceҲ��������������ʹ���˵��µ����ܲ�׼�����⡣ģ�͵�������ʹ��benchmark�������������ݼ��������õ�bs1��bs16����������Ϊ׼������ʹ��benchmark���߲��Ե�batch4��8��32������������README.md����������¼���ɡ�  
1.benchmark�������������ݼ������������������  
batch1�����ܣ�benchmark�������������ݼ�������������result/perf_vision_batchsize_1_device_0.txt��  
```
[e2e] throughputRate: 126.32, latency: 395819
[data read] throughputRate: 134.323, moduleLatency: 7.44476
[preprocess] throughputRate: 133.845, moduleLatency: 7.47134
[infer] throughputRate: 127.16, Interface throughputRate: 180.131, moduleLatency: 6.90735
[post] throughputRate: 127.159, moduleLatency: 7.86415
```
Interface throughputRate: 180.131��180.131x4=720.524����batch1 310����������  

batch16�����ܣ�benchmark�������������ݼ�������������result/perf_vision_batchsize_16_device_1.txt��  
```
[e2e] throughputRate: 156.123, latency: 320261
[data read] throughputRate: 165.096, moduleLatency: 6.05708
[preprocess] throughputRate: 164.628, moduleLatency: 6.07431
[infer] throughputRate: 156.862, Interface throughputRate: 291.04, moduleLatency: 5.1523
[post] throughputRate: 9.80371, moduleLatency: 102.002
```
Interface throughputRate: 291.04��291.04x4=1,164.16����batch16 310����������  


./benchmark.x86_64 -batch_size=4 -om_path=./model_rectify_random.onnx.om -round=50 -device_id=0
batch4�����ܣ�benchmark���ߴ�����������result/PureInfer_perf_of_resnet152_bs4_in_device_0.txt��  
```

ave_throughputRate = 244.742samples/s, ave_latency = 4.15733ms

```
Interface throughputRate: 244.742��244.742x4=978.968����batch4 310���������� 

batch8�����ܣ�benchmark���߽��д���������result/PureInfer_perf_of_resnet152_bs8_in_device_0.txt��  
```
ave_throughputRate = 270.962samples/s, ave_latency = 3.75835ms

```
Interface throughputRate: 270.962��270.962x4=1,083.848����batch8 310����������   

batch32�����ܣ�benchmark���ߴ�����������result/PureInfer_perf_of_resnet152_bs32_in_device_0.txt��  
```
ave_throughputRate = 270.402samples/s, ave_latency = 3.71981ms

```
Interface throughputRate: 270.402��270.402x4=1,081.608����batch32 310����������  

### 7.2 T4��������
��װ��T4���ķ������ϲ���gpu���ܣ����Թ�����ȷ����û��������������TensorRT�汾��7.2.3.4��cuda�汾��11.0��cudnn�汾��8.2  
batch1���ܣ�
```
trtexec --onnx=resnet152.onnx --fp16 --shapes=image:1x3x224x224 --threads
```
gpu T4��4��device����ִ�еĽ����mean��ʱ�ӣ�tensorrt��ʱ����batch�����ݵ�����ʱ�䣩���������ʵĵ�������batch
```
[06/11/2021-02:43:51] [I] GPU Compute
[06/11/2021-02:43:51] [I] min: 2.9082 ms
[06/11/2021-02:43:51] [I] max: 6.05182 ms
[06/11/2021-02:43:51] [I] mean: 3.00433 ms
[06/11/2021-02:43:51] [I] median: 2.97778 ms
[06/11/2021-02:43:51] [I] percentile: 3.16479 ms at 99%
[06/11/2021-02:43:51] [I] total compute time: 3.00133 s
```
batch1 t4���������ʣ�1000/(3.00433/1)=332.8529156251144fps 
```
 

batch16���ܣ�
```
trtexec --onnx=resnet152.onnx --fp16 --shapes=image:16x3x224x224 --threads
```
[06/11/2021-02:50:44] [I] GPU Compute
[06/11/2021-02:50:44] [I] min: 19.9592 ms
[06/11/2021-02:50:44] [I] max: 22.4021 ms
[06/11/2021-02:50:44] [I] mean: 21.0969 ms
[06/11/2021-02:50:44] [I] median: 20.9503 ms
[06/11/2021-02:50:44] [I] percentile: 22.3171 ms at 99%
[06/11/2021-02:50:44] [I] total compute time: 3.03795 s
```
batch16 t4���������ʣ�1000/(21.0969/16)=758.4052633325275fps  


batch4���ܣ�
```
[06/11/2021-08:01:43] [I] GPU Compute
[06/11/2021-08:01:43] [I] min: 6.2175 ms
[06/11/2021-08:01:43] [I] max: 12.7552 ms
[06/11/2021-08:01:43] [I] mean: 6.57256 ms
[06/11/2021-08:01:43] [I] median: 6.47629 ms
[06/11/2021-08:01:43] [I] percentile: 6.98999 ms at 99%
[06/11/2021-08:01:43] [I] total compute time: 3.01023 s

```
batch4 t4���������ʣ�1000/(6.57256/4)=608.590868702606fps 



batch8���ܣ�
```
[06/11/2021-08:03:59] [I] GPU Compute
[06/11/2021-08:03:59] [I] min: 10.8062 ms
[06/11/2021-08:03:59] [I] max: 12.296 ms
[06/11/2021-08:03:59] [I] mean: 11.3813 ms
[06/11/2021-08:03:59] [I] median: 11.2798 ms
[06/11/2021-08:03:59] [I] percentile: 12.2863 ms at 99%
[06/11/2021-08:03:59] [I] total compute time: 3.02744 s

```
batch8 t4���������ʣ�1000/(11.3813/8)=702.9074007362955fps 



batch32���ܣ�
```
[06/11/2021-08:06:39] [I] GPU Compute
[06/11/2021-08:06:39] [I] min: 39.5345 ms
[06/11/2021-08:06:39] [I] max: 52.6029 ms
[06/11/2021-08:06:39] [I] mean: 44.5667 ms
[06/11/2021-08:06:39] [I] median: 43.1216 ms
[06/11/2021-08:06:39] [I] percentile: 52.6029 ms at 99%
[06/11/2021-08:06:39] [I] total compute time: 3.0751 s

```
batch32 t4���������ʣ�1000/(44.5667/32)=718.0248930255101fps 

```

### 7.3 ���ܶԱ�
batch1��180.131x4 > 1000/(3.00433/1)  
batch16��291.04x4 > 1000/(21.0969/16)  
310����device�������ʳ�4�����������ʱ�T4�����������ʴ󣬹�310���ܸ���T4���ܣ����ܴ�ꡣ  
����batch1��310���ܸ���T4����2.16����batch16��310���ܸ���T4����1.535������ģ�ͷ���Benchmark/cv/classificationĿ¼�¡�  
 **�����Ż���**  
>û���������ܲ��������⣬�ʲ���Ҫ���������Ż�

