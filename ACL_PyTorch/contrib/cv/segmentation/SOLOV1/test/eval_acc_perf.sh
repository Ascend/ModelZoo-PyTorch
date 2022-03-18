#!/bin/bash
datasets_path="/root/datasets"
for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done
arch=`uname -m`
rm -rf ./val2017_bin
rm -rf ./val2017_bin_meta
python solov1_preprocess.py \
  --image_src_path=${datasets_path}/coco/val2017 \
  --bin_file_path=val2017_bin \
  --meta_file_path=val2017_bin_meta \
  --model_input_height=800 \
  --model_input_width=1216
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python get_info.py ${datasets_path}/coco/ \
  SOLO/configs/solo/solo_r50_fpn_8gpu_1x.py \
  val2017_bin \
  val2017_bin_meta \
  solo.info \
  solo_meta.info \
  1216 800
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

source env.sh
rm -rf result/dumpOutput_device0
./benchmark.${arch} -model_type=vision -om_path=solo.om -device_id=0 -batch_size=1 -input_text_path=solo.info \
  -input_width=1216 -input_height=800 -useDvpp=false -output_binary=true
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python solov1_postprocess.py \
  --dataset_path=${datasets_path}/coco/ \
  --model_config=SOLO/configs/solo/solo_r50_fpn_8gpu_1x.py \
  --bin_data_path=./result/dumpOutput_device0/ \
  --meta_info=solo_meta.info \
  --net_out_num=3 \
  --model_input_height 800 \
  --model_input_width 1216
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

echo "====performance data===="
python test/parse.py result/perf_vision_batchsize_1_device_0.txt
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi
echo "success"