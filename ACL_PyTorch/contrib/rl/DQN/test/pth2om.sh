echo "====om transform begin===="
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf dqn_bs1.om
atc --framework=5 --model=dqn.onnx --output=dqn_bs1 --input_format=NCHW --input_shape="input:1,4,84,84"  --log=error --soc_version=Ascend310  --op_select_implmode=high_performance

if [ -f "dqn_bs1.om" ];  then
    echo "success"
else
    echo "fail!"
fi

