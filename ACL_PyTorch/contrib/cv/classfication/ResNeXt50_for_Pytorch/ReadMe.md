 **ResNeXt-50模型PyTorch离线推理指导** 

1 模型概述
论文地址 https://arxiv.org/abs/1611.05431
代码地址 https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
输入输出数据 
输入数据 大小 				 数据类型 数据排布格式
input    batchsizex3x224x224 RGB_FP32 NCHW
输出数据 大小     数据类型 数据排布格式
output1  1 x 1000 FLOAT32  ND

2 环境说明
2.1 深度学习框架
ONNX 1.7.0
Pytorch 1.6.0
TorchVision 0.7.0
2.2 python第三方库
numpy 1.18.5
Pillow 7.2.0
	
3 数据集预处理
3.1 获取原始数据集
本模型支持ImageNet 50000张图片的验证集。图片与标签分别存放在/root/datasets/imagenet/val与/root/datasets/imagenet/val_label.txt。
使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签
3.2 数据集预处理
执行preprocess_resnext50_pth.py脚本处理数据集，生成数据集预处理后的bin文件
python3 preprocess_resnext50_pth.py dataset/ImageNet/val_union/ pre_bin
3.3 生成数据集信息文件
执行get_info.py脚本，生成推理输入的数据集二进制info文件
python3 get_info.py bin pre_bin resnext50_val.info 224 224

4 pth转onnx模型
4.1 下载pth权重文件
从链接https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth下载
4.2 生成onnx模型文件
执行resnext50_pth2onnx.py脚本转化为onnx模型
python3.7 resnext50_pth2onnx.py ./resnext50_32x4d-7cdf4587.pth ./resnext50.onnx
4.3 onnx转om模型
（1）修改resnext50_atc.sh脚本，通过ATC工具使用脚本完成转换,具体的脚本示例如下：
	# 使用二进制输入时，执行如下命令
	atc --model=./resnext50.onnx --framework=5 --output=resnext50_bs16 --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --log=info --soc_version=Ascend${chip_name}
	# 使用JPEG输入时，执行如下命令
	atc --model=./resnext50.onnx --framework=5 --output=resnext50_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,224,224" --log=info --soc_version=Ascend${chip_name} --insert_op_conf=aipp_resnext50_pth.config

（2）运行resnext50_atc.sh脚本转换om模型
	bash resnext50_atc.sh

5 离线推理
5.1 安装ais_infer工具
https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer
5.2 使用ais_infer离线推理
python ais_infer.py --model "/home/zzy/resnext50_bs16_310.om" --input /home/zzy/prep_bin/  --output "/home/zzy/output/" --outfmt  TXT  --batchsize 16
5.3 验证推理结果，第一个参数为生成推理结果所在路径，第二个参数为标签数据，第三个参数为生成结果文件路径，第四个参数为生成结果文件名称。
python3.7 vision_metric_ImageNet.py output/2022_08_04-17_21_14/ ./val_label.txt ./ result.json
	

6 模型推理精度
		 top1   top5
310精度  77.61% 93.68%
310p精度 77.62% 93.69%
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。
精度调试：
没有遇到精度不达标的问题，故不需要进行精度调试

7 模型推理精度性能
    310p    T4     310  310p/310 310p/T4
bs1 1413.46 759.14 642.98 2.1982 1.8619
bs4 2886.57 1028.36 1248.2 2.3125 2.8069
bs8 3503.03 1183.76 1530.728 2.2884 2.9592
bs16 3861.76 1259.39 2070.524 1.8651 3.0663
bs32 4012.15 1354.95 1612.724 2.4878 2.9611
bs64 1985.7 1451.07 1492.348 1.3305 1.3684
最优bs 4012.15 1451.07 2070.524 1.9377 2.7649
最优batch:
710大于310的1.2;710大于T4的1.6倍，性能达标
性能优化：
没有遇到性能不达标的问题，故不需要进行性能优化