 #!/bin/bash
arch=`uname -m`
echo $arch

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

echo "====data pre_treatment starting===="
rm -rf ./prep_dataset
python3.7.5 ADNet_preprocess.py ./dataset/BSD68 ./prep_dataset
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====data pre_treatment finished===="


echo "====creating info file===="
rm -rf ./ADNet_prep_bin.info
python3.7.5 gen_dataset_info.py ./prep_dataset/INoisy ADNet_prep_bin.info 481 321 
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====creating info file done===="


echo "====bs1 benchmark start===="
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=./ADNet_bs1.om -input_text_path=./ADNet_prep_bin.info -input_width=481 -input_height=321 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====bs1 benchmark finished===="


echo "====bs16 benchmark start===="
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result/dumpOutput_device1
./benchmark.${arch} -model_type=vision -device_id=1 -batch_size=16 -om_path=./ADNet_bs16.om -input_text_path=./ADNet_prep_bin.info -input_width=481 -input_height=321 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====bs16 benchmark finished===="


echo "====bs1 evaluate start; this costs a little time===="
rm -rf ADNet_bs1.log
python3.7.5 -u ADNet_postprocess.py ./result/dumpOutput_device0 ./prep_dataset/ISoure ./out >ADNet_bs1.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====bs1 evaluate finished===="


echo "====bs16 evaluate start; this costs a little time===="
rm -rf ADNet_bs16.log
python3.7.5 -u ADNet_postprocess.py ./result/dumpOutput_device1 ./prep_dataset/ISoure ./out >ADNet_bs16.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====bs16 evaluate finished===="


echo "====accuracy data===="
echo "pth psnr:29.25"
python3.7.5 test/parse.py ADNet_bs1.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


python3.7.5 test/parse.py ADNet_bs16.log
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi


echo "====performance data===="
python3.7 test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"


python3.7.5 test/parse.py result/perf_vision_batchsize_16_device_1.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"
