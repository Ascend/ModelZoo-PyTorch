python DnCNN_pth2onnx.py net.pth DnCNN-S-15.onnx  #执行pth2onnx脚本，生成onnx模型文件
source env.sh  #设置环境变量
atc --framework=5 --model=./DnCNN-S-15.onnx --input_format=NCHW --input_shape="actual_input_1:1,1,481,481" --output=DnCNN-S-15_bs1 --log=debug --soc_version=Ascend710