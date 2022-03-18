#!/bin/bash

datasets_path="Chinese-Text-Classification-Pytorch/THUCNews/"

for para in $*
do
	if [[ $para == --datasets_path* ]]; then
		datasets_path=`echo ${para#*=}`
	fi
done

echo $datasets_path

arch=`uname -m`
# cd ..
rm -rf ./prep_dataset_query
python3.7 TextCNN_preprocess.py --dataset ${datasets_path} --save_folder ./prep_dataset_query
python3.7 gen_dataset_info.py ./prep_dataset_query dataset.info
if [ $? != 0 ]; then
	echo "fail!"
	exit -1
fi

source env.sh
rm -rf result/dumpOutput_device0
rm -rf result/dumpOutput_device0_bs1
rm -rf result/dumpOutput_device0_bs16

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest

./benchmark.x86_64 -model_type=nlp -device_id=0 -batch_size=1 -om_path=TextCNN_bs1.om -output_binary=True -input_text_path=dataset.info -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
mv result/dumpOutput_device0 result/dumpOutput_device0_bs1

./benchmark.x86_64 -model_type=nlp -device_id=0 -batch_size=16 -om_path=TextCNN_bs16.om -output_binary=True -input_text_path=dataset.info -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
mv result/dumpOutput_device0 result/dumpOutput_device0_bs16

python3.7 TextCNN_postprocess.py result/dumpOutput_device0_bs1 > result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 TextCNN_postprocess.py result/dumpOutput_device0_bs16 > result_bs16.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====accuracy data===="
python3.7 test/parse.py result_bs1.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py result_bs16.json
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "====performance data===="
python3.7 test/parse.py result/perf_nlp_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
python3.7 test/parse.py result/perf_nlp_batchsize_16_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"

