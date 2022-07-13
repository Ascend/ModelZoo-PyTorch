export chip_name=$1

rm -rf spach_ms_conv_s.onnx

python3 ./SPACH_pth2onnx.py --input-path="spach_ms_conv_s.pth" --output-path="spach_ms_conv_s.onnx"

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

rm -rf spach_ms_conv_s_1.om spach_ms_conv_s_16.om

source /usr/local/Ascend/ascend-toolkit/set_env.sh

for bs in {1,16}
do
   export TUNE_BANK_PATH=./custom_tune_bank/bs${bs}
   atc --model=spach_ms_conv_s.onnx --framework=5 --output=spach_ms_conv_s_${bs} --op_precision_mode=op_precision.ini --log=error --input_format=NCHW --input_shape="input:${bs},3,224,224" --output_type=FP16 --soc_version=${chip_name}
done


if [ -f "spach_ms_conv_s_1.om" ] && [ -f "spach_ms_conv_s_16.om" ]; then
    echo "success"
else
    echo "fail!"
fi
