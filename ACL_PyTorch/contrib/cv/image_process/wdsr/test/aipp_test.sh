#!/bin/bash
. /usr/local/Ascend/ascend-toolkit/set_env.sh  
rm -rf wdsr.onnx
python3.7 Wdsr_pth2onnx.py --ckpt epoch_30.pth --model wdsr --output_name wdsr.onnx --scale 2

rm -rf Wdsr_bs1.om 
atc --framework=5 --model=wdsr.onnx --output=wdsr_bs1 --input_format=NCHW --input_shape="image:1,3,1020,1020" --log=debug --soc_version=Ascend310

datasets_path="/root/datasets/div2k"

rm -rf ./DIV2K_valid_LR_bicubic_bin/X2/
python3.7 Wdsr_prePorcess.py --lr_path ${datasets_path}/DIV2K_valid_LR_bicubic/X2/ --hr_path ${datasets_path}/HR/ --save_lr_path ./LR/  --width 1020 --height 1020 --scale 2

python3.7 get_info.py bin /root/datasets/div2k/LR/ wdsr_bin.info 1020 1020

chmod u+x benchmark.x86_64
rm -rf result/dumpOutput_device0
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./wdsr_bs1.om -input_text_path=./wdsr_bin.info -input_width=1020 -input_height=1020 -output_binary=True -useDvpp=False

echo "====accuracy data===="
python3.7 Wdsr_postProcess.py --bin_data_path ./result/dumpOutput_device0/ --dataset_path ${datasets_path}/HR/ --result result_bs1.txt --scale 2

echo "====performance data===="
python3.7 test/parse.py result/perf_vision_batchsize_1_device_0.txt