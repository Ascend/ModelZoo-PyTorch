datasets_path="./Set5"
for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
rm -rf pre_dataset
python3.7.5 CSNLN_preprocess.py --s ${datasets_path}/LR_bicubic/X4/ --d prep_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf prep_bin.info
python3.7.5 get_info.py bin prep_dataset/bin_56 prep_bin.info 56 56
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -batch_size=1 -device_id=0 -om_path=csnln_x4_bs1.om -input_text_path=prep_bin.info -input_width=56 -input_height=56 -useDvpp=false -output_binary=true
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
rm -rf result_bs1.log
python3.7.5 CSNLN_postprocess.py --hr ${datasets_path}/HR/ --res result/dumpOutput_device0 --save_path res_png > result_bs1.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====accuracy data===="
python3.7.5 test/parse.py result_bs1.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data===="
python3.7.5 test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"
