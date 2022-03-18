#!/bin/bash
datasets_path="/root/datasets/imagenet/val"
val_label_path=""
for para in $@
do
    if [[ $para == --datasets_path* ]];then
        datasets_path=`echo ${para#*=}`
    elif [[ $para == --val_label_path* ]];then
        val_label_path=`echo ${para#*=}`
    fi
done
echo datasets_path:${datasets_path}
echo val_label_path:${val_label_path}
# 校验是否传入data_path,不需要修改
if [[ $val_label_path == "" ]];then
    echo "[Error] para \"val_label_path\" must be confing"
    exit 1
fi

python3 TResNet_preprocess.py ${datasets_path} ./prep_dataset
python3 gen_dataset_info.py bin ./prep_dataset ./tresnet_prep_bin.info 224 224
bs=(1 16)
for i in ${bs[@]}; do
  ./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=$i -om_path=tresnet_patch16_224_bs${i}.om -input_text_path=./tresnet_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
  python3 TResNet_postprocess.py result/dumpOutput_device0/ $val_label_path ./ result${i}.json
done
