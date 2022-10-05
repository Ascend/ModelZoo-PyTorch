rm -rf fairmot.onnx
python3.7 fairmot_pth2onnx.py --input_file=fairmot_dla34.pth --output_file=fairmot.onnx

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf fairmot_bs1.om fairmot_bs8.om
atc --framework=5 --model=./fairmot.onnx --input_format=NCHW --input_shape="actual_input_1:1,3,608,1088" --output=./fairmot_bs1 --log=debug --soc_version=Ascend310 
atc --framework=5 --model=./fairmot.onnx --input_format=NCHW --input_shape="actual_input_1:8,3,608,1088" --output=./fairmot_bs8 --log=debug --soc_version=Ascend310 
if [ -f "fairmot_bs1.om" ] && [ -f "fairmot_bs8.om" ]; then
    echo "success"
else
    echo "fail!"
fi
