#!/bin/bash
#  bash ./test/eval_acc_perf.sh --datasets_path='./datasets/facades'

source /usr/local/Ascend/ascend-toolkit/set_env.sh
for para in $*
do
    if [[ $para == --datasets_path* ]];then
        datasets_path=`echo ${para#*=}`
    fi
done


for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

arch=`uname -m`
python pix2pix_preprocess.py \
    --dataroot ${datasets_path} \



python gen_dataset_info.py bin ./datasets/facades/bin ./netG_prep_bin.info 256 256



rm -rf result/dumpOutput_device0
rm -rf result/dumpOutput_device0_bs1
rm -rf result/dumpOutput_device0_bs16

./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=./checkpoints/facades_label2photo_pretrained/netG_om_bs1.om -input_text_path=./netG_prep_bin.info -input_width=256 -input_height=256 -output_binary=True -useDvpp=False

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

mv result/dumpOutput_device0 result/dumpOutput_device0_bs1

./benchmark.${arch} -model_type=vision -device_id=0 -batch_size=16 -om_path=./checkpoints/facades_label2photo_pretrained/netG_om_bs16.om -input_text_path=./netG_prep_bin.info -input_width=256 -input_height=256 -output_binary=True -useDvpp=False
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

mv result/dumpOutput_device0 result/dumpOutput_device0_bs16

python3.7 pix2pix_postprocess.py --bin2img_file=./result/bin2img_bs1/  --npu_bin_file=./result/dumpOutput_device0_bs1/

python3.7 pix2pix_postprocess.py --bin2img_file=./result/bin2img_bs16/  --npu_bin_file=./result/dumpOutput_device0_bs16/

# if [ $? != 0 ]; then
#     echo "fail!"
#     exit -1
# fi

# python3.7 ReID_postprocess.py --query_dir=${datasets_path}/market1501/query --gallery_dir=${datasets_path}/market1501/bounding_box_test --pred_dir=./result/dumpOutput_device0_bs16 > result_bs16.json

# if [ $? != 0 ]; then
#     echo "fail!"
#     exit -1
# fi

# echo "====accuracy data===="

# python3.7 test/parse.py result_bs1.json
# if [ $? != 0 ]; then
#     echo "fail!"
#     exit -1
# fi

# python3.7 test/parse.py result_bs16.json
# if [ $? != 0 ]; then
#     echo "fail!"
#     exit -1
# fi

# echo "====performance data===="
# python3.7 test/parse.py result/perf_vision_batchsize_1_device_0.txt
# if [ $? != 0 ]; then
#     echo "fail!"
#     exit -1
# fi
# python3.7 test/parse.py result/perf_vision_batchsize_16_device_0.txt
# if [ $? != 0 ]; then
#     echo "fail!"
#     exit -1
# fi
# echo "success"