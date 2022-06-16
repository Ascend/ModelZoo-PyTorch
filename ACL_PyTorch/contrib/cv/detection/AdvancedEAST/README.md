# AdvancedEAST模型PyTorch离线推理指导

## 1 环境准备

1.安装必要的依赖
```
pip3.7 install -r requirements.txt
```

2.获取开源模型代码
```
git clone https://github.com/BaoWentz/AdvancedEAST-PyTorch -b master
cd AdvancedEAST-PyTorch
git reset a835c8cedce4ada1bc9580754245183d9f4aaa17 --hard
cd ..  
```

3.获取权重文件

[AdvancedEAST预训练pth权重文件](https://pan.baidu.com/s/1NSyc-cHKV3IwDo6qojIrKA)，密码: ye9y

解压后使用3T736_best_mF1_score.pth，文件sha1: 9D0C603C4AA4E955FEA04925F3E01E793FEF4045

4.获取数据集

[天池ICPR数据集](https://pan.baidu.com/s/1NSyc-cHKV3IwDo6qojIrKA)，密码: ye9y

下载ICPR_text_train_part2_20180313.zip和[update] ICPR_text_train_part1_20180316.zip两个压缩包，新建目录icpr和子目录icpr/image_10000、icpr/txt_10000，将压缩包中image_9000、image_1000中的图片文件解压至image_10000中，将压缩包中txt_9000、txt_1000中的标签文件解压至txt_10000中

5.[获取benchmark工具](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/)

将benchmark.x86_64或benchmark.aarch64放到当前目录

## 2 数据预处理

1.图片缩放、标签转换为npy文件和图片转换为bin文件
	python3.7 AdvancedEAST_preprocess.py icpr prep_dataset
	第一个参数为数据集的路径，第二个参数为生成bin文件的路径。

2.生成数据集info文件
	python3.7 gen_dataset_info.py bin prep_dataset prep_bin.info 736 736
	第一个参数为生成的数据集文件格式，第二个参数为预处理后的bin文件的路径，第三个参数为生成的数据集文件保存的路径，第四第五个参数为图片的宽和高。运行成功后，在当前目录中生成prep_bin.info。

## 3 离线推理 

1.模型转换
	1)将模型权重文件.pth转换为.onnx文件
		python3.7 AdvancedEAST_pth2onnx.py 3T736_best_mF1_score.pth AdvancedEAST_dybs.onnx
	
	2)设置atc工作所需要的环境变量
		source /usr/local/Ascend/ascend-toolkit/set_env.sh
	
	3)将.onnx文件转为离线推理模型文件.om文件
		atc --framework=5 --model=AdvancedEAST_dybs.onnx --output=AdvancedEAST_bs1 --input_format=NCHW --input_shape='input_1:1,3,736,736' --log=debug --soc_version=Ascend${chip_name}
		参数说明：
		--model：为ONNX模型文件。
		--framework：5代表ONNX模型。
		--output：输出的OM模型。
		--input_format：输入数据的格式。
		--input_shape：输入数据的shape。
		--log：日志级别。
		--soc_version：处理器型号。

2.开始推理
	1）增加benchmark.{arch}可执行权限
		chmod u+x benchmark.x86_64
		
	2）推理
		310P上执行，执行时使npu-smi info查看设备状态，确保device空闲
		
		./benchmark.x86_64 -model_type=vision -batch_size=1 -device_id=0 -om_path=AdvancedEAST_bs1.om -input_text_path=prep_bin.info -input_width=736 -input_height=736 -useDvpp=false -output_binary=true
		参数说明：
		-- model_type：模型的类型。
		-- batch_size：执行一次模型推理所处理的数据量。
		-- device_id：运行的Device编号。
		-- om_path：经过ATC转换后的模型OM文件所在的路径。
		-- input_text_path：模型对应的数据集所在的路径。
		-- input_width：输入模型的宽度。
		-- input_height：输入模型的高度。
		-- useDvpp：模型前处理是否使用DVPP编解码模块。
		-- output_binary：输出结果格式是否为二进制文件（即bin文件）。
		
	3）精度验证
		python3.7 AdvancedEAST_postprocess.py icpr result/dumpOutput_device0
		第一个参数为数据集路径，第二个参数为推理结果所在路径。

**评测结果：**
| 模型      | 官网pth精度  | 310离线推理精度  | 基准性能    | 310P性能    |
| :------: | :------: | :------: | :------:  | :------:  |
| AdvancedEAST bs1  | f1-score:52.08% | f1-score:52.08% | 84.626fps | 137.91fps |
| AdvancedEAST bs16 | f1-score:52.08% | f1-score:52.08% | 86.304fps | 95.5733fps |

