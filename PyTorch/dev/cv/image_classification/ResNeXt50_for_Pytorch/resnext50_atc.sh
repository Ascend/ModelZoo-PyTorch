# 使用二进制输入时，执行如下命令
atc --model=./resnext50.onnx --framework=5 --output=resnext50_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,224,224" --log=info --soc_version=Ascend310

atc --model=./resnext50.onnx --framework=5 --output=resnext50_bs4 --input_format=NCHW --input_shape="actual_input_1:4,3,224,224" --log=info --soc_version=Ascend310

atc --model=./resnext50.onnx --framework=5 --output=resnext50_bs8 --input_format=NCHW --input_shape="actual_input_1:8,3,224,224" --log=info --soc_version=Ascend310

atc --model=./resnext50.onnx --framework=5 --output=resnext50_bs16 --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --log=info --soc_version=Ascend310

atc --model=./resnext50.onnx --framework=5 --output=resnext50_bs32 --input_format=NCHW --input_shape="actual_input_1:32,3,224,224" --log=info --soc_version=Ascend310

atc --model=./resnext50.onnx --framework=5 --output=resnext50_bs64 --input_format=NCHW --input_shape="actual_input_1:64,3,224,224" --log=info --soc_version=Ascend310
