#!/bin/bash
set -eu

arch=`arch`

# preprocess
python3.7 preprocess.py --image_dir './datasets/ECSSD/images' --save_dir './test_data_ECSSD'
python3.7 gen_dataset_info.py bin ./test_data_ECSSD ECSSD.info 320 320

# infer process
source env.sh
chmod a+x benchmark.${arch}
device_id=0
./benchmark.${arch} -model_type=vision -device_id=$device_id -batch_size=1 -om_path=./models/u2net_sim_bs1_fixv2.om -input_text_path=./ECSSD.info -input_width=320 -input_height=320 -output_binary=True -useDvpp=False
mv ./result/dumpOutput_device0 ./result/bs1
./benchmark.${arch} -model_type=vision -device_id=$device_id -batch_size=16 -om_path=./models/u2net_sim_bs16_fixv2.om -input_text_path=./ECSSD.info -input_width=320 -input_height=320 -output_binary=True -useDvpp=False
mv ./result/dumpOutput_device0 ./result/bs16


# postprocess
python3.7 postprocess.py --image_dir ./datasets/ECSSD/images --save_dir ./test_vis_ECSSD_bs1 --out_dir ./result/bs1
python3.7 postprocess.py --image_dir ./datasets/ECSSD/images --save_dir ./test_vis_ECSSD_bs16 --out_dir ./result/bs16


# evaluation
echo 'bs1 evaluation result:'
python3.7 evaluate.py --res_dir ./test_vis_ECSSD_bs1 --gt_dir ./datasets/ECSSD/ground_truth_mask
echo 'bs16 evaluation result:'
python3.7 evaluate.py --res_dir ./test_vis_ECSSD_bs16 --gt_dir ./datasets/ECSSD/ground_truth_mask
