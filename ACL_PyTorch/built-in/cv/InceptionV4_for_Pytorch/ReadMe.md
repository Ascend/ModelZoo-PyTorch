# 推理环境准备： #

1.onnx==1.7.0

2.pytorch==1.5.0

3.torchvision==0.6.0

4.numpy==1.18.5

5.pillow==7.2.0

6.python==3.7.5



# 文件作用说明： #

1.inceptionv4_pth2onnx.py：用于转换pth模型文件到onnx模型文件

2.inceptionv4_atc.sh：onnx模型转换om模型脚本

3.preprocess_inceptionv4_pth.py：数据集预处理脚本，通过均值方差处理归一化图片，生成图片二进制文件

4.aipp_inceptionv4_pth.config：数据集aipp预处理配置文件

5.get_info.py：生成推理输入的数据集二进制info文件或jpg info文件

6.inceptionv4_val.info：ImageNet验证集二进制info文件，用于benchmark推理获取数据集

7.ImageNet.info：ImageNet验证集jpg info文件，用于benchmark推理获取数据集

8.val_label.txt：ImageNet数据集标签，用于验证推理结果

9.rename_result.py: 对使用ais\_infer.py推理得到的结果进行重命名，便于后续验证推理结果

10.vision_metric_ImageNet.py：验证推理结果脚本，比对benchmark输出的分类结果和标签，给出Accuracy





# 推理端到端步骤： #

**（1）获取权重文件**

  开源[Link](https://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth)在PyTorch开源预训练模型中获取inceptionv4权重文件

**（2） 从https://github.com/Cadene/pretrained-models.pytorch下载inceptionv4模型**
	
	步骤：
	（a）进入到InceptionV4模型文件夹中，命令如下：
	cd /home/HwHiAiUser/ATC_InceptionV4_from_Pytorch_Ascend310/InceptionV4_Pytorch_Infer/

	（b）clone pretrainedmodels代码库
	git clone https://github.com/Cadene/pretrained-models.pytorch.git

	（c） 将pretrainedmodels代码拉下来后，需要将pretrained-models.pytorch/pretrainedmodels/models/inception4.py 中的
		adaptiveAvgPoolWidth = features.shape[2] 
	修改为
		adaptiveAvgPoolWidth = features.shape[2].item
	（d）安装pretrainedmodels，命令如下
  		cd pretrained-models.pytorch
 		python3 setup.py install





**（3）导出onnx文件**
	
通过inceptionv4_pth2onnx.py脚本将.pth文件转化为onnx模型，执行如下命令：

	Python3.7 inceptionv4_pth2onnx.py ./inceptionv4-8e4777a0.pth ./inception4.onnx

第一个参数为输入权重文件路径，第二个参数为输出onnx文件路径。
运行成功后，在当前目录生成inceptinv4.onnx模型文件。



**（4）运行inceptionv4_atc.sh脚本将onnx文件转换为om模型**

（4.1） 修改inceptionv4_atc.sh脚本，通过ATC工具使用脚本完成转换，具体的脚本示例如下：

	# 配置环境变量
	export install_path=/usr/local/Ascend/ascend-toolkit/latest
	export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
	export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
	export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
	export ASCEND_OPP_PATH=${install_path}/opp
	batch_size=$1
	chip_name=$2
	# 使用二进制输入时，执行如下命令
	atc --model=./inceptionv4.onnx --framework=5 --output=inceptionv4_bs${batch_size} --input_format=NCHW --input_shape="actual_input_1:${batch_size},3,299,299" --log=info --soc_version=Ascend${chip_name}
    
参数说明：

--model：为ONNX模型文件。

--framework：5代表ONNX模型。

--output：输出的OM模型。

--input_format：输入数据的格式。

--input_shape：输入数据的shape


(4.2) 执行inceptionv4_atc.sh脚本，将.onnx文件转为离线推理模型文件.om文件，执行一下命令

     ./inceptionv4_atc.sh ${batch_size} ${chip_name}
	注意，${batch_size}为批大小 ${chip_name}可通过命令 npu-smi info得到，
	如果chip_name为710，batchsize为16，则命令为./inceptionv4_atc.sh 16 710，运行成功后在生成inceptionv4_bs16.om
	如果chip_name为310，batchsize为8，则命令为./inceptionv4_atc.sh 8 310，运行成功后在生成inceptionv4_bs8.om



**（5）执行preprocess_inceptionv4_pth.py脚本**

将原始数据（.JPEG）转化为二进制文件（.bin）。转化方法参考pretrainedmodels训练预处理方法处理数据，以获得最佳精度。通过缩放、均值方差手段归一化，输出为二进制文件。

	python3 preprocess_inceptionv4_pth.py /home/HwHiAiUser/dataset/ImageNet/ILSVRC2012_img_val ./prep_bin

第一个参数为原始数据验证集（.JPEG）所在路径，第二个参数为输出的二进制文件（.bin）所在路径。每个图像对应生成一个二进制文件
	

**(6) 使用ais_infer推理工具进行推理**

	（a）安装 ais_infer 推理工具
		安装步骤请见：https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer
	（b）运行ais_infer.py进行推理
		python3.7 /home/zzy_zhao/tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model ./inceptionv4_bs16.om --batchsize 16 --input ./prep_bin --output ais_infer_txt_result --outfmt TXT
		运行ais_infer推理，结果保存在 ./ais_infer_txt_result 目录
		参数说明：
		--model：om模型文件。
		--batchsize：批大小。
		--input：二进制文件（.bin）所在路径。
		--output：推理结果保存路径。
		--outfmt：推理结果保存格式。


**（7）验证推理结果**

因为使用ais\_infer推理，生成的txt文件和sumary.json文件中，故需要将sumary.json文件删除，便可调用vision_metric_ImageNet.py脚本与数据集标签val_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。具体步骤如下

	
	（a）删除由ais_infer离线推理生成的sumary.json文件
     rm -rf ./ais_infer_txt_result/2022_07_19-10_39_38/sumary.json
    （b）使用vision_metric_ImageNet.py 验证推理结果
    python3.7 vision_metric_ImageNet.py ./ais_infer_txt_result/2022_07_19-10_39_38/ ./val_label.txt ./ result.json
   


