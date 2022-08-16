#!/bin/bash
#  bash ./eval_acc_perf.sh --datasets_path='./datasets/facades'

source /usr/local/Ascend/ascend-toolkit/set_env.sh

for para in $*
do
    if [[ $para == --datasets_path* ]];then
        datasets_path=`echo ${para#*=}`
    fi
done

python pix2pix_preprocess.py --dataroot ${datasets_path} 

for para in $*
do
    if [[ $para == --datasets_path* ]]; then
        datasets_path=`echo ${para#*=}`
    fi
done

rm -rf result/dumpOutput_device0
rm -rf result/dumpOutput_device0_bs1
rm -rf result/dumpOutput_device0_bs4
rm -rf result/dumpOutput_device0_bs8
rm -rf result/dumpOutput_device0_bs16
rm -rf result/dumpOutput_device0_bs32
rm -rf result/dumpOutput_device0_bs64

mkdir -p result
mkdir result/dumpOutput_device0
mkdir result/dumpOutput_device0_bs1
mkdir result/dumpOutput_device0_bs4
mkdir result/dumpOutput_device0_bs8
mkdir result/dumpOutput_device0_bs16
mkdir result/dumpOutput_device0_bs32
mkdir result/dumpOutput_device0_bs64

python ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model ./checkpoints/facades_label2photo_pretrained/netG_om_bs1.om --input  "./datasets/facades/bin" --output "result/dumpOutput_device0_bs1" --outfmt BIN --batchsize 1

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model ./checkpoints/facades_label2photo_pretrained/netG_om_bs4.om --input  "./datasets/facades/bin" --output "result/dumpOutput_device0_bs4" --outfmt BIN --batchsize 4

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model ./checkpoints/facades_label2photo_pretrained/netG_om_bs8.om --input  "./datasets/facades/bin" --output "result/dumpOutput_device0_bs8" --outfmt BIN --batchsize 8

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model ./checkpoints/facades_label2photo_pretrained/netG_om_bs16.om --input  "./datasets/facades/bin" --output "result/dumpOutput_device0_bs16" --outfmt BIN --batchsize 16

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model ./checkpoints/facades_label2photo_pretrained/netG_om_bs32.om --input  "./datasets/facades/bin" --output "result/dumpOutput_device0_bs32" --outfmt BIN --batchsize 32

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

python ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model ./checkpoints/facades_label2photo_pretrained/netG_om_bs64.om --input  "./datasets/facades/bin" --output "result/dumpOutput_device0_bs64" --outfmt BIN --batchsize 64

if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi

result_path=./result/dumpOutput_device0_bs1/
files=$(ls $result_path)
for filename in $files
do
    res_bin_file=$result_path$filename/
done
python3.7 pix2pix_postprocess.py --bin2img_file=./result/bin2img_bs1/  --npu_bin_file=$res_bin_file

result_path=./result/dumpOutput_device0_bs4/
files=$(ls $result_path)
for filename in $files
do
    res_bin_file=$result_path$filename/
done
python3.7 pix2pix_postprocess.py --bin2img_file=./result/bin2img_bs4/  --npu_bin_file=$res_bin_file

result_path=./result/dumpOutput_device0_bs8/
files=$(ls $result_path)
for filename in $files
do
    res_bin_file=$result_path$filename/
done
python3.7 pix2pix_postprocess.py --bin2img_file=./result/bin2img_bs8/  --npu_bin_file=$res_bin_file

result_path=./result/dumpOutput_device0_bs16/
files=$(ls $result_path)
for filename in $files
do
    res_bin_file=$result_path$filename/
done
python3.7 pix2pix_postprocess.py --bin2img_file=./result/bin2img_bs16/  --npu_bin_file=$res_bin_file

result_path=./result/dumpOutput_device0_bs32/
files=$(ls $result_path)
for filename in $files
do
    res_bin_file=$result_path$filename/
done
python3.7 pix2pix_postprocess.py --bin2img_file=./result/bin2img_bs32/  --npu_bin_file=$res_bin_file

result_path=./result/dumpOutput_device0_bs64/
files=$(ls $result_path)
for filename in $files
do
    res_bin_file=$result_path$filename/
done
python3.7 pix2pix_postprocess.py --bin2img_file=./result/bin2img_bs64/  --npu_bin_file=$res_bin_file
