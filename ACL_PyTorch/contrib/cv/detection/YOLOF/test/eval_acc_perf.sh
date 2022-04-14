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
--meta_file_path val2017_bin_meta \
--batch_size ${batch_size}

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python gen_dataset_info.py \
--meta_file_path val2017_bin_meta \
--meta_info_file_name yolof_meta.info

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf result

./msame --model "yolof.om" --input "val2017_bin" --output "result" --outfmt BIN

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python YOLOF_postprocess.py \
--pth_path YOLOF_CSP_D_53_DC5_9x.pth \
--bin_data_path result/ \
--meta_info_path yolof_meta.info \

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

