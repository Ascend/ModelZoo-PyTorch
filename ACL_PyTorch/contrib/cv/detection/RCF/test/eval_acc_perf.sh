#!/bin/bash

datasets_path="data/BSR/BSDS500/data/images/test"
batch_size=1
device_id=2

# usage
if [ $# -ne 6 ]
then
    echo "usage: bash test/eval_acc_perf.sh datasets_path data/BSR/BSDS500/data/images/test batch_size 1 device_id 2, \
Now use the default parameters"
else
    datasets_path=$2
    batch_size=$4
    device_id=$6
fi

arch=`uname -m`
python3.7 rcf_preprocess.py --src_dir=${datasets_path} --save_name=./data/images_bin --height 321 481 --width 481 321
if [ $? != 0 ]; then
    echo "preprocess fail!"
    exit -1
else
    echo "preprocess success!"
fi

python3.7 gen_dataset_info.py bin data/images_bin_321x481 rcf_prep_bin_321x481.info 481 321
python3.7 gen_dataset_info.py bin data/images_bin_481x321 rcf_prep_bin_481x321.info 321 481
if [ $? != 0 ]; then
    echo "gen dataset info fail!"
    exit -1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result
chmod +x benchmark.${arch}
./benchmark.x86_64 -model_type=vision -device_id=${device_id} -batch_size=${batch_size} -om_path=rcf_bs${batch_size}_321x481.om \
-input_text_path=./rcf_prep_bin_321x481.info -input_width=481 -input_height=321 -output_binary=True -useDvpp=False
mv result/perf_vision_batchsize_${batch_size}_device_${device_id}.txt result/perf_vision_batchsize_${batch_size}_device_${device_id}_321x481.txt
./benchmark.x86_64 -model_type=vision -device_id=${device_id} -batch_size=${batch_size} -om_path=rcf_bs${batch_size}_481x321.om \
-input_text_path=./rcf_prep_bin_481x321.info -input_width=321 -input_height=481 -output_binary=True -useDvpp=False
mv result/perf_vision_batchsize_${batch_size}_device_${device_id}.txt result/perf_vision_batchsize_${batch_size}_device_${device_id}_481x321.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python3.7 rcf_postprocess.py --model om --imgs_dir data/BSR/BSDS500/data/images/test \
--bin_dir result/dumpOutput_device${device_id} --om_output data/om_out
if [ $? != 0 ]; then
    echo "post process fail!"
    exit -1
else
    echo "post process success!"
fi

cd edge_eval_python
cd cxx/src
# source build.sh
cd ../..
rm -rf ../data/examples_om/rcf_eval_result
mkdir -p ../data/examples_om/rcf_eval_result
python3.7 main.py --alg "RCF" --model_name_list "rcf" --result_dir ../data/om_out \
--save_dir ../data/examples_om/rcf_eval_result --gt_dir ../data/BSR/BSDS500/data/groundTruth/test \
--key om_result --file_format .mat --workers -1
cd ..
if [ $? != 0 ]; then
    echo "evaluate fail!"
    exit -1
else
    echo "evaluate success!"
fi

python3.7 test/parse.py --file_type txt --file_name result/perf_vision_batchsize_${batch_size}_device_${device_id}_321x481.txt \
result/perf_vision_batchsize_${batch_size}_device_${device_id}_481x321.txt --bin_path data/images_bin_321x481 data/images_bin_481x321
