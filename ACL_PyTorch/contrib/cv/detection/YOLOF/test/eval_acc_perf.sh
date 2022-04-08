#!/bin/bash

set -eu

batch_size=1

for para in $*
do
    if [[ $para == --batch_size* ]]; then
        batch_size=`echo ${para#*=}`
    fi
done


arch=`uname -m`

rm -rf val2017_bin
rm -rf val2017_bin_meta
python YOLOF_preprocess.py \
--bin_file_path val2017_bin \
--meta_file_path val2017_bin_meta

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python gen_dataset_info.py \
--bin_file_path val2017_bin \
--meta_file_path val2017_bin_meta \
--bin_info_file_name yolof.info \
--meta_info_file_name yolof_meta.info

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result

./benchmark.${arch} -model_type=vision -om_path=yolof.om -device_id=0 -batch_size=${batch_size} \
-input_text_path=yolof.info -input_width=608 -input_height=608 -useDvpp=false -output_binary=true

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python YOLOF_postprocess.py \
--pth_path YOLOF_CSP_D_53_DC5_9x.pth \
--bin_data_path result/dumpOutput_device0/ \
--meta_info_path yolof_meta.info \

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====performance data===="
python test/parse.py result/perf_vision_batchsize_${batch_size}_device_0.txt

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"

