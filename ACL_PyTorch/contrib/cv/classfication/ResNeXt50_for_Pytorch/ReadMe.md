文件作用说明：

1.preprocess_resnext50_pth.py：数据集预处理脚本，通过均值方差处理归一化图片，生成图片二进制文件

2.get_info.py：生成推理输入的数据集二进制info文件或jpg info文件

3.aipp_resnext50_pth.config：数据集aipp预处理配置文件

4.resnext50_pth2onnx.py：用于转换pth模型文件到onnx模型文件

5.resnext50_atc.sh：onnx模型转换om模型脚本

6.ais_infer.py：离线推理工具

7.vision_metric_ImageNet.py：验证推理结果脚本，比对benchmark输出的分类结果和标签，给出Accuracy



推理端到端步骤：

（1）用preprocess_resnext50_pth.py脚本处理数据集，参考resnext50_val.info配置处理后的二进制数据集路径。或者配置数据集aipp预处理文件aipp_resnext50_pth.config。
    python3 preprocess_resnext50_pth.py dataset/ImageNet/val_union/ pre_bin


（2）生成推理输入的数据集二进制info文件
    python3 get_info.py bin pre_bin resnext50_val.info 224 224

（3）从Torchvision下载resnext50模型，通过resnext50_pth2onnx.py脚本转化为onnx模型

	python3.7 resnext50_pth2onnx.py ./resnext50_32x4d-7cdf4587.pth ./resnext50.onnx


（4）修改resnext50_atc.sh脚本，通过ATC工具使用脚本完成转换,具体的脚本示例如下：
	# 配置环境变量
	export install_path=/usr/local/Ascend/ascend-toolkit/latest
	export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
	export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
	export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
	export ASCEND_OPP_PATH=${install_path}/opp
	# 使用二进制输入时，执行如下命令
	atc --model=./resnext50.onnx --framework=5 --output=resnext50_bs16 --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --log=info --soc_version=Ascend${chip_name}
	# 使用JPEG输入时，执行如下命令
	atc --model=./resnext50.onnx --framework=5 --output=resnext50_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,224,224" --log=info --soc_version=Ascend${chip_name} --insert_op_conf=aipp_resnext50_pth.config

（5）运行resnext50_atc.sh脚本转换om模型
	bash resnext50_atc.sh

本demo已提供调优完成的om模型

	
（6）使用ais_infer离线推理
	python ais_infer.py --model "/home/zzy/resnext50_bs16_310.om" --input /home/zzy/prep_bin/  --output "/home/zzy/output/" --outfmt  TXT  --batchsize 16
	
	
（7）验证推理结果，第一个参数为生成推理结果所在路径，第二个参数为标签数据，第三个参数为生成结果文件路径，第四个参数为生成结果文件名称。
	python3.7 vision_metric_ImageNet.py output/2022_08_04-17_21_14/ ./val_label.txt ./ result.json
	


